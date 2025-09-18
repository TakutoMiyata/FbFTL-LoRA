"""
CIFAR-optimized ResNet with LoRA for FedSA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from cifar_resnet import CIFARResNet, BasicBlock, Bottleneck


class LoRALayer(nn.Module):
    """LoRA adapter layer for CIFAR ResNet with separate A and B parameters"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1, seed=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0
        
        # Only create LoRA matrices if rank > 0
        if rank > 0:
            # A matrix: shared across clients with DP
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            # B matrix: kept local per client
            self.lora_B = nn.Parameter(torch.empty(out_features, rank))
            
            self.dropout = nn.Dropout(dropout)
            
            # Initialize with proper seeding for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # LoRA paper recommended initialization
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            # No LoRA when rank=0
            self.lora_A = None
            self.lora_B = None
            self.dropout = None
    
    def forward(self, x):
        # If no LoRA (rank=0), return zeros
        if self.rank == 0:
            original_shape = x.shape
            output_shape = original_shape[:-1] + (self.out_features,)
            return torch.zeros(output_shape, dtype=x.dtype, device=x.device)
        
        # LoRA forward
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # Apply dropout and LoRA transformation
        x_dropout = self.dropout(x_flat)
        lora_intermediate = torch.matmul(x_dropout, self.lora_A.t())
        lora_out = torch.matmul(lora_intermediate, self.lora_B.t()) * self.scaling
        
        # Reshape back
        new_shape = original_shape[:-1] + (self.out_features,)
        return lora_out.view(new_shape)


class CIFARResNetWithLoRA(nn.Module):
    """CIFAR ResNet model with LoRA adapters for federated learning"""
    
    def __init__(self, config):
        super().__init__()
        
        # Load configuration
        self.model_name = config.get('model_name', 'resnet18')
        self.num_classes = config.get('num_classes', 100)
        self.lora_r = config.get('lora_r', 8)
        self.lora_alpha = config.get('lora_alpha', 16)
        self.lora_dropout = config.get('lora_dropout', 0.1)
        self.seed = config.get('seed', None)
        
        # Create base CIFAR ResNet model
        if self.model_name == 'resnet18':
            self.base_model = CIFARResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.num_classes)
        elif self.model_name == 'resnet34':
            self.base_model = CIFARResNet(BasicBlock, [3, 4, 6, 3], num_classes=self.num_classes)
        elif self.model_name == 'resnet50':
            self.base_model = CIFARResNet(Bottleneck, [3, 4, 6, 3], num_classes=self.num_classes)
        elif self.model_name == 'resnet101':
            self.base_model = CIFARResNet(Bottleneck, [3, 4, 23, 3], num_classes=self.num_classes)
        elif self.model_name == 'resnet152':
            self.base_model = CIFARResNet(Bottleneck, [3, 8, 36, 3], num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported ResNet model: {self.model_name}")
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get the number of features in the final layer
        num_features = self.base_model.fc.in_features
        
        # Replace classifier with trainable one
        self.base_model.fc = nn.Identity()  # Remove original fc
        self.classifier = nn.Linear(num_features, self.num_classes)
        
        # Add LoRA adapters to selected layers (if lora_r > 0)
        self.lora_adapters = nn.ModuleDict()
        
        if self.lora_r > 0:
            # Add LoRA to the last residual blocks of layer4
            if self.model_name in ['resnet50', 'resnet101', 'resnet152']:
                # Bottleneck blocks
                for block_idx in range(len(self.base_model.layer4)):
                    # Add LoRA after conv3 (1x1 conv that expands channels)
                    self.lora_adapters[f'layer4_{block_idx}_conv3'] = LoRALayer(
                        512 * 4, 512 * 4,  # Bottleneck expansion = 4
                        rank=self.lora_r,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                        seed=self.seed + block_idx if self.seed else None
                    )
            else:
                # Basic blocks (ResNet18/34)
                for block_idx in range(len(self.base_model.layer4)):
                    # Add LoRA after conv2
                    self.lora_adapters[f'layer4_{block_idx}_conv2'] = LoRALayer(
                        512, 512,  # Basic block, no expansion
                        rank=self.lora_r,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                        seed=self.seed + block_idx if self.seed else None
                    )
            
            # Add LoRA to classifier (optional)
            self.classifier_lora = LoRALayer(
                num_features, self.num_classes,
                rank=self.lora_r,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                seed=self.seed + 100 if self.seed else None
            )
        else:
            self.classifier_lora = None
    
    def forward_with_lora(self, x):
        """Forward pass with LoRA applied to layer4"""
        # Initial convolution
        out = F.relu(self.base_model.bn1(self.base_model.conv1(x)))
        
        # First three layers (no LoRA)
        out = self.base_model.layer1(out)
        out = self.base_model.layer2(out)
        out = self.base_model.layer3(out)
        
        # Layer 4 with LoRA
        for block_idx, block in enumerate(self.base_model.layer4):
            identity = out
            
            if isinstance(block, Bottleneck):
                # Bottleneck block
                out = F.relu(block.bn1(block.conv1(out)))
                out = F.relu(block.bn2(block.conv2(out)))
                out = block.bn3(block.conv3(out))
                
                # Apply LoRA after conv3
                lora_key = f'layer4_{block_idx}_conv3'
                if lora_key in self.lora_adapters:
                    b, c, h, w = out.shape
                    out_reshaped = out.permute(0, 2, 3, 1).reshape(b * h * w, c)
                    lora_out = self.lora_adapters[lora_key](out_reshaped)
                    lora_out = lora_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
                    out = out + lora_out
            else:
                # Basic block
                out = F.relu(block.bn1(block.conv1(out)))
                out = block.bn2(block.conv2(out))
                
                # Apply LoRA after conv2
                lora_key = f'layer4_{block_idx}_conv2'
                if lora_key in self.lora_adapters:
                    b, c, h, w = out.shape
                    out_reshaped = out.permute(0, 2, 3, 1).reshape(b * h * w, c)
                    lora_out = self.lora_adapters[lora_key](out_reshaped)
                    lora_out = lora_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
                    out = out + lora_out
            
            # Shortcut connection
            if hasattr(block, 'shortcut') and len(block.shortcut) > 0:
                identity = block.shortcut(identity)
            
            out += identity
            out = F.relu(out)
        
        # Global average pooling
        out = self.base_model.avgpool(out)
        features = torch.flatten(out, 1)
        
        # Classification with LoRA
        logits = self.classifier(features)
        if self.classifier_lora is not None:
            logits = logits + self.classifier_lora(features)
        
        return logits
    
    def forward(self, x):
        """Standard forward pass"""
        if self.lora_r > 0:
            return self.forward_with_lora(x)
        else:
            # No LoRA, just use base model with new classifier
            features = self.base_model.extract_features(x)
            return self.classifier(features)
    
    def extract_features(self, x):
        """Extract features before classifier"""
        return self.base_model.extract_features(x)
    
    def get_A_parameters(self) -> Dict[str, torch.Tensor]:
        """Get A matrices for FedSA (shared across clients)"""
        A_params = {}
        
        if self.lora_r > 0:
            for name, adapter in self.lora_adapters.items():
                if adapter.lora_A is not None:
                    A_params[f'{name}_A'] = adapter.lora_A
            
            if self.classifier_lora is not None and self.classifier_lora.lora_A is not None:
                A_params['classifier_lora_A'] = self.classifier_lora.lora_A
        
        return A_params
    
    def get_B_parameters(self) -> Dict[str, torch.Tensor]:
        """Get B matrices (kept local per client)"""
        B_params = {}
        
        if self.lora_r > 0:
            for name, adapter in self.lora_adapters.items():
                if adapter.lora_B is not None:
                    B_params[f'{name}_B'] = adapter.lora_B
            
            if self.classifier_lora is not None and self.classifier_lora.lora_B is not None:
                B_params['classifier_lora_B'] = self.classifier_lora.lora_B
        
        # Include classifier parameters as local
        B_params['classifier.weight'] = self.classifier.weight
        B_params['classifier.bias'] = self.classifier.bias
        
        return B_params
    
    def set_A_parameters(self, A_params: Dict[str, torch.Tensor]):
        """Set A matrices from server"""
        for name, param_value in A_params.items():
            if name == 'classifier_lora_A':
                if self.classifier_lora is not None:
                    self.classifier_lora.lora_A.data.copy_(param_value.data)
            else:
                # Extract adapter name
                adapter_name = name.replace('_A', '')
                if adapter_name in self.lora_adapters:
                    self.lora_adapters[adapter_name].lora_A.data.copy_(param_value.data)
    
    def get_A_parameter_groups(self):
        """Get A parameters as a list for optimizer"""
        params = []
        if self.lora_r > 0:
            for adapter in self.lora_adapters.values():
                if adapter.lora_A is not None:
                    params.append(adapter.lora_A)
            if self.classifier_lora is not None and self.classifier_lora.lora_A is not None:
                params.append(self.classifier_lora.lora_A)
        return params
    
    def get_B_parameter_groups(self):
        """Get B parameters as a list for optimizer"""
        params = []
        if self.lora_r > 0:
            for adapter in self.lora_adapters.values():
                if adapter.lora_B is not None:
                    params.append(adapter.lora_B)
            if self.classifier_lora is not None and self.classifier_lora.lora_B is not None:
                params.append(self.classifier_lora.lora_B)
        
        # Include classifier parameters
        params.extend([self.classifier.weight, self.classifier.bias])
        return params
    
    def get_parameter_groups_for_optimizer(self, lr_scale_for_B=1.0):
        """Get parameter groups for optimizer with optional different LR for B"""
        param_groups = []
        
        # A parameters (if using LoRA)
        A_params = self.get_A_parameter_groups()
        if A_params:
            param_groups.append({'params': A_params, 'name': 'lora_A'})
        
        # B parameters and classifier
        B_params = self.get_B_parameter_groups()
        if B_params:
            param_groups.append({
                'params': B_params,
                'name': 'lora_B_and_classifier',
                'lr_scale': lr_scale_for_B
            })
        
        return param_groups if param_groups else [{'params': self.parameters()}]


def create_cifar_resnet_lora(config):
    """Create CIFAR ResNet with LoRA for FedSA"""
    return CIFARResNetWithLoRA(config)