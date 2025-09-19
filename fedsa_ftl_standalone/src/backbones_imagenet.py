"""
ImageNet pre-trained backbones with LoRA injection for federated transfer learning
Supports MobileNetV2, EfficientNet-B0, and ResNet variants
"""

import math
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


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
        # Explicitly create as float32 for AMP compatibility
        self.lora_A = nn.Conv2d(in_c, r, kernel_size=1, bias=False).float()
        self.lora_B = nn.Conv2d(r, out_c, kernel_size=1, bias=False).float()
        
        self.scaling = alpha / float(r) if r > 0 else 0
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights (LoRA paper recommendations)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze base convolution
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Save input dtype for AMP compatibility
        input_dtype = x.dtype
        
        # Base convolution output (uses original dtype)
        base_out = self.base(x)
        
        # LoRA computation always in float32 to avoid AMP issues
        lora_out = self.lora_B(self.lora_A(x.float()))
        lora_out = self.dropout(lora_out) * self.scaling
        
        # Convert LoRA output back to input dtype and add to base
        return base_out + lora_out.to(input_dtype)
    
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
                    module.lora_A.weight.data = A_params[param_name].clone()
    
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
        
        # Keep trainable: classifier/fc layers and LoRA parameters
        if any(key in name for key in ["classifier", "fc", "lora_A", "lora_B"]):
            p.requires_grad = True
            trainable_params += p.numel()
        else:
            p.requires_grad = False
            frozen_params += p.numel()
    
    # Freeze BatchNorm layers (disable training mode for stability)
    bn_count = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            bn_count += 1
    
    if verbose:
        print(f"🔒 Backbone freezing:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"  BatchNorm layers frozen: {bn_count}")


# ---------- Backbone factory ----------
def build_backbone(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Build ImageNet pre-trained backbone model
    
    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pre-trained weights
    
    Returns:
        PyTorch model
    """
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # Replace classifier (torchvision >= 0.13 structure)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        # Replace fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: mobilenet_v2, efficientnet_b0, resnet18, resnet50")


def make_model_with_lora(config: Dict[str, Any]):
    """
    Create model with optional LoRA injection based on configuration
    
    Args:
        config: Configuration dictionary containing model settings
    
    Returns:
        PyTorch model with optional LoRA adapters
    """
    # Extract model configuration
    model_config = config.get('model', {})
    model_name = model_config.get('model_name', 'mobilenet_v2')
    num_classes = model_config.get('num_classes', 100)
    pretrained = model_config.get('pretrained', True)
    freeze_backbone = model_config.get('freeze_backbone', True)
    
    # LoRA configuration
    lora_config = model_config.get('lora', {})
    use_lora = lora_config.get('enabled', False)
    lora_r = lora_config.get('r', 4)
    lora_alpha = lora_config.get('alpha', 8)
    lora_dropout = lora_config.get('dropout', 0.1)
    
    print(f"🏗️  Building {model_name} (pretrained={pretrained}, classes={num_classes})")
    
    # Build base model
    model = build_backbone(model_name, num_classes, pretrained)
    
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


def get_model_info(model: nn.Module):
    """
    Get detailed information about model parameters
    
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


def print_model_summary(config: Dict[str, Any], model: nn.Module):
    """Print detailed model summary for debugging and verification"""
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    info = get_model_info(model)
    
    print("📊 Model Summary:")
    print(f"  Input size: {data_config.get('input_size', 224)}")
    print(f"  Pretrained: {model_config.get('pretrained', True)}")
    print(f"  Backbone: {model_config.get('model_name', 'unknown')}")
    print(f"  LoRA enabled: {model_config.get('lora', {}).get('enabled', False)}")
    print(f"  Trainable params (M): {info['trainable_params']/1e6:.2f}")
    print(f"  Total params (M): {info['total_params']/1e6:.2f}")
    print(f"  Compression ratio: {1/info['trainable_ratio']:.1f}x" if info['trainable_ratio'] > 0 else "N/A")
    
    if info['lora_params'] > 0:
        print(f"  LoRA params (M): {info['lora_params']/1e6:.2f}")
        print(f"  LoRA ratio: {info['lora_ratio']*100:.1f}%")