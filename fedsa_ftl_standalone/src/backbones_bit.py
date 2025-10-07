"""
Big Transfer (BiT) models with LoRA injection for federated transfer learning
Supports BiT-S and BiT-M variants with ResNet-50 and ResNet-101
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any

# Import timm for BiT models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


# ---------- LoRA modules ----------
class LoRAConv2d(nn.Module):
    """LoRA adapter for Conv2d layers (1x1 pointwise convolutions)"""

    def __init__(self, base_conv: nn.Conv2d, r=4, alpha=8, dropout=0.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        assert base_conv.kernel_size == (1, 1), "LoRA only applied to 1x1 convolutions"

        self.base = base_conv
        in_c, out_c = base_conv.in_channels, base_conv.out_channels

        # LoRA matrices: A (down-projection) and B (up-projection)
        self.lora_A = nn.Conv2d(in_c, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, out_c, kernel_size=1, bias=False)

        self.scaling = alpha / float(r) if r > 0 else 0
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

        # Initialize LoRA weights (LoRA paper recommendations)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base convolution
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Base convolution output
        base_out = self.base(x)

        # LoRA computation
        lora_out = self.lora_B(self.lora_A(x))
        lora_out = self.dropout(lora_out) * self.scaling

        return base_out + lora_out

    def to(self, *args, **kwargs):
        """Ensure LoRA layers are moved to the same device and dtype as the base layer"""
        result = super().to(*args, **kwargs)
        # Make sure LoRA layers match the base layer's device and dtype
        if hasattr(result, 'base') and hasattr(result, 'lora_A') and hasattr(result, 'lora_B'):
            target_device = result.base.weight.device
            target_dtype = result.base.weight.dtype
            result.lora_A = result.lora_A.to(device=target_device, dtype=target_dtype)
            result.lora_B = result.lora_B.to(device=target_device, dtype=target_dtype)
        return result

    def get_lora_parameters(self):
        """Get LoRA-specific parameters for federated learning"""
        return {
            'lora_A': self.lora_A.weight,
            'lora_B': self.lora_B.weight
        }


def inject_lora_to_pointwise_convs(model: nn.Module, r=4, alpha=8, dropout=0.0):
    """
    Inject LoRA adapters to all 1x1 convolutions in the model

    Args:
        model: PyTorch model to modify
        r: LoRA rank
        alpha: LoRA scaling parameter
        dropout: Dropout rate for LoRA layers

    Returns:
        Number of LoRA layers injected
    """
    lora_count = 0

    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if (isinstance(child, nn.Conv2d) and
                child.kernel_size == (1, 1) and
                child.groups == 1):

                # Replace with LoRA version
                lora_conv = LoRAConv2d(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_conv)
                lora_count += 1

    print(f"✅ Injected LoRA into {lora_count} pointwise convolutions (r={r}, alpha={alpha})")
    return lora_count


def add_lora_methods_to_model(model: nn.Module):
    """
    Add LoRA parameter methods to the model for FedSA-LoRA compatibility

    Args:
        model: PyTorch model with LoRA adapters
    """
    def get_A_parameters(self):
        """Get all LoRA A matrix parameters for FedSA-LoRA aggregation"""
        A_params = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRAConv2d):
                A_params[f"{name}.lora_A.weight"] = module.lora_A.weight.data.clone()
        return A_params

    def get_B_parameters(self):
        """Get all LoRA B matrix parameters"""
        B_params = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRAConv2d):
                B_params[f"{name}.lora_B.weight"] = module.lora_B.weight.data.clone()
        return B_params

    def set_A_parameters(self, A_params):
        """Set LoRA A matrix parameters from aggregated values"""
        for name, module in self.named_modules():
            if isinstance(module, LoRAConv2d):
                param_name = f"{name}.lora_A.weight"
                if param_name in A_params:
                    # Ensure the parameter is on the same device and dtype as the current weight
                    device = module.lora_A.weight.device
                    dtype = module.lora_A.weight.dtype
                    # Use copy_ to update in-place instead of creating new tensor
                    # This preserves the parameter identity for optimizer
                    module.lora_A.weight.data.copy_(A_params[param_name].to(device=device, dtype=dtype))

    def get_A_parameter_groups(self):
        """Get LoRA A matrix parameters as a list for optimizer"""
        A_params = []
        for name, module in self.named_modules():
            if isinstance(module, LoRAConv2d):
                A_params.append(module.lora_A.weight)
        return A_params

    def get_B_parameter_groups(self):
        """Get LoRA B matrix parameters as a list for optimizer"""
        B_params = []
        for name, module in self.named_modules():
            if isinstance(module, LoRAConv2d):
                B_params.append(module.lora_B.weight)
        return B_params

    # Bind methods to model instance
    import types
    model.get_A_parameters = types.MethodType(get_A_parameters, model)
    model.get_B_parameters = types.MethodType(get_B_parameters, model)
    model.set_A_parameters = types.MethodType(set_A_parameters, model)
    model.get_A_parameter_groups = types.MethodType(get_A_parameter_groups, model)
    model.get_B_parameter_groups = types.MethodType(get_B_parameter_groups, model)


def freeze_backbone_except_head_and_lora(model: nn.Module, verbose=True):
    """
    Freeze backbone parameters except classifier head and LoRA adapters
    Also freeze BatchNorm layers for transfer learning stability

    Args:
        model: PyTorch model to freeze
        verbose: Whether to print freezing statistics
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, p in model.named_parameters():
        total_params += p.numel()

        # Keep trainable: classifier/fc/head layers and LoRA parameters
        if any(key in name for key in ["classifier", "fc", "head", "lora_A", "lora_B"]):
            p.requires_grad = True
            trainable_params += p.numel()
        else:
            p.requires_grad = False
            frozen_params += p.numel()

    # Freeze BatchNorm layers (disable training mode for stability)
    bn_count = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            bn_count += 1

    if verbose:
        print(f"🔒 Backbone freezing:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"  Normalization layers frozen: {bn_count}")


# ---------- BiT model factory ----------
def build_bit_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Build Big Transfer (BiT) model using timm

    Args:
        model_name: BiT model variant
            - 'bit_s_r50x1': BiT-S ResNet-50×1
            - 'bit_m_r50x1': BiT-M ResNet-50×1
            - 'bit_s_r101x1': BiT-S ResNet-101×1
            - 'bit_m_r101x1': BiT-M ResNet-101×1
        num_classes: Number of output classes
        pretrained: Whether to use BiT pre-trained weights

    Returns:
        PyTorch model
    """
    if not TIMM_AVAILABLE:
        raise RuntimeError(
            "timm is required for BiT models. "
            "Install with: pip install timm"
        )

    # Map model names to timm model identifiers
    model_mapping = {
        'bit_s_r50x1': 'resnetv2_50x1_bit.goog_in21k',  # BiT-S (ImageNet-21k)
        'bit_m_r50x1': 'resnetv2_50x1_bit.goog_in21k_ft_in1k',  # BiT-M (ImageNet-21k + ImageNet-1k fine-tuned)
        'bit_s_r101x1': 'resnetv2_101x1_bit.goog_in21k',  # BiT-S ResNet-101
        'bit_m_r101x1': 'resnetv2_101x1_bit.goog_in21k_ft_in1k',  # BiT-M ResNet-101
    }

    if model_name not in model_mapping:
        raise ValueError(
            f"Unknown BiT model: {model_name}. "
            f"Supported: {list(model_mapping.keys())}"
        )

    timm_model_name = model_mapping[model_name]

    # Create model with timm
    print(f"🏗️  Loading BiT model: {timm_model_name}")
    model = timm.create_model(
        timm_model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model


def make_bit_model_with_lora(config: Dict[str, Any]):
    """
    Create BiT model with optional LoRA injection based on configuration

    Args:
        config: Configuration dictionary containing model settings

    Returns:
        PyTorch model with optional LoRA adapters
    """
    # Extract model configuration
    model_config = config.get('model', {})
    model_name = model_config.get('model_name', 'bit_m_r50x1')
    num_classes = model_config.get('num_classes', 100)
    pretrained = model_config.get('pretrained', True)
    freeze_backbone = model_config.get('freeze_backbone', True)

    # LoRA configuration
    lora_config = model_config.get('lora', {})
    use_lora = lora_config.get('enabled', False)
    lora_r = lora_config.get('r', 4)
    lora_alpha = lora_config.get('alpha', 8)
    lora_dropout = lora_config.get('dropout', 0.1)

    print(f"🏗️  Building BiT model: {model_name} (pretrained={pretrained}, classes={num_classes})")

    # Build BiT model
    model = build_bit_model(model_name, num_classes, pretrained)

    # Debug: Check if pretrained weights are loaded
    if pretrained:
        first_param = next(model.parameters())
        weight_mean = first_param.data.abs().mean().item()
        print(f"✅ Pretrained weights check: First layer weight mean = {weight_mean:.6f}")
        if weight_mean < 0.001:
            print("⚠️  WARNING: Weights appear to be near zero! Pretrained weights may not be loaded correctly!")
        else:
            print("✅ Pretrained weights loaded successfully")

    # Inject LoRA if enabled
    if use_lora:
        print(f"🔧 Injecting LoRA adapters...")
        inject_lora_to_pointwise_convs(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        # Add FedSA-LoRA methods for A matrix parameter handling
        add_lora_methods_to_model(model)

    # Freeze backbone if requested
    if freeze_backbone:
        freeze_backbone_except_head_and_lora(model, verbose=True)

    return model


def get_bit_model_info(model: nn.Module):
    """
    Get detailed information about BiT model parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Count LoRA parameters
    lora_params = 0
    for name, p in model.named_parameters():
        if 'lora_' in name and p.requires_grad:
            lora_params += p.numel()

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'lora_params': lora_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'lora_ratio': lora_params / total_params if total_params > 0 else 0
    }


def print_bit_model_summary(config: Dict[str, Any], model: nn.Module):
    """Print detailed BiT model summary for debugging and verification"""
    model_config = config.get('model', {})
    data_config = config.get('data', {})

    info = get_bit_model_info(model)

    print("📊 BiT Model Summary:")
    print(f"  Model variant: {model_config.get('model_name', 'unknown')}")
    print(f"  Input size: {data_config.get('input_size', 224)}")
    print(f"  Pretrained: {model_config.get('pretrained', True)}")
    print(f"  LoRA enabled: {model_config.get('lora', {}).get('enabled', False)}")
    print(f"  Total params (M): {info['total_params']/1e6:.2f}")
    print(f"  Trainable params (M): {info['trainable_params']/1e6:.2f}")
    print(f"  Compression ratio: {1/info['trainable_ratio']:.1f}x" if info['trainable_ratio'] > 0 else "N/A")

    if info['lora_params'] > 0:
        print(f"  LoRA params (M): {info['lora_params']/1e6:.2f}")
        print(f"  LoRA ratio: {info['lora_ratio']*100:.1f}%")
