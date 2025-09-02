"""
FedSA-FTL Model Implementation
VGG16 backbone with LoRA adaptation following FbFTL paper approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
import math


class FedSAFTLModel(nn.Module):
    """
    FedSA-FTL Model with frozen VGG16 backbone and LoRA-adapted head
    Following FbFTL paper approach with ImageNet pre-trained VGG16
    """
    
    def __init__(self, num_classes=10, model_name='vgg16', 
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, freeze_backbone=True):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # Initialize VGG16 backbone
        self._init_vgg_backbone(model_name, num_classes)
        
        # Store LoRA configuration
        self.lora_config = {
            'r': lora_r,
            'alpha': lora_alpha,
            'dropout': lora_dropout
        }
        
        # Store num_classes
        self.num_classes = num_classes
        
        # Apply LoRA to classifier layers
        self._apply_lora_to_classifier()
    
    
    def _init_vgg_backbone(self, model_name, num_classes):
        """Initialize VGG backbone"""
        print(f"Loading pre-trained VGG model: {model_name}")
        
        # Load pre-trained VGG
        if model_name == 'vgg16':
            self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg16_bn':
            self.backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported VGG variant: {model_name}")
        
        # Get feature dimension and remove original classifier
        feature_dim = self.backbone.classifier[0].in_features  # 25088
        self.backbone.classifier = nn.Identity()
        
        # Add adaptive pooling to handle CIFAR-10 32x32 input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Create classification head with reduced capacity for CIFAR
        # Original VGG classifier is too large for CIFAR datasets
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),  # Reduced from 4096
            nn.ReLU(True),
            nn.Dropout(0.2),  # Reduced dropout
            nn.Linear(512, 512),  # Reduced from 4096
            nn.ReLU(True), 
            nn.Dropout(0.2),  # Reduced dropout
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if specified (following FbFTL approach)
        if self.freeze_backbone:
            print("Freezing VGG backbone parameters...")
            for param in self.backbone.features.parameters():
                param.requires_grad = False
    
    def _apply_lora_to_classifier(self):
        """Apply LoRA decomposition to linear layers in classifier"""
        for i, module in enumerate(self.classifier):
            if isinstance(module, nn.Linear):
                # Get original weights
                original_weight = module.weight.data.clone()
                original_bias = module.bias.data.clone() if module.bias is not None else None
                
                # Replace with LoRA linear layer
                self.classifier[i] = LoRALinear(
                    module.in_features,
                    module.out_features,
                    r=self.lora_config['r'],
                    lora_alpha=self.lora_config['alpha'],
                    lora_dropout=self.lora_config['dropout'],
                    original_weight=original_weight,
                    original_bias=original_bias
                )
    
    def forward(self, x):
        """Forward pass for VGG16"""
        # Extract features with frozen VGG backbone
        # The backbone is already frozen via requires_grad=False, so we don't need torch.set_grad_enabled
        features = self.backbone.features(x)
        features = self.adaptive_pool(features)
        features = torch.flatten(features, 1)
        
        # Detach features to prevent gradients flowing to frozen backbone
        if self.freeze_backbone:
            features = features.detach()
        
        # Pass through LoRA-adapted classifier
        logits = self.classifier(features)
        return logits
    
    def get_lora_params(self, matrix_type='both'):
        """
        Get LoRA parameters
        Args:
            matrix_type: 'A', 'B', or 'both'
        Returns:
            Dictionary of LoRA parameters
        """
        lora_params = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                if matrix_type in ['A', 'both']:
                    lora_params[f"{name}.lora_A"] = module.lora_A.data.clone()
                if matrix_type in ['B', 'both']:
                    lora_params[f"{name}.lora_B"] = module.lora_B.data.clone()
        return lora_params
    
    def set_lora_params(self, lora_params, matrix_type='both'):
        """
        Set LoRA parameters
        Args:
            lora_params: Dictionary of LoRA parameters
            matrix_type: 'A', 'B', or 'both'
        """
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                if matrix_type in ['A', 'both'] and f"{name}.lora_A" in lora_params:
                    module.lora_A.data.copy_(lora_params[f"{name}.lora_A"])
                if matrix_type in ['B', 'both'] and f"{name}.lora_B" in lora_params:
                    module.lora_B.data.copy_(lora_params[f"{name}.lora_B"])


class LoRALinear(nn.Module):
    """
    LoRA-adapted Linear layer
    W = W0 + BA where W0 is frozen, B and A are low-rank matrices
    """
    
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.1,
                 original_weight=None, original_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Frozen pre-trained weights
        if original_weight is not None:
            self.weight = nn.Parameter(original_weight.clone())
        else:
            # Fallback to random initialization if no pre-trained weights available
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        if original_bias is not None:
            self.bias = nn.Parameter(original_bias.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Freeze original parameters
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.dropout = nn.Dropout(lora_dropout)
        
        # Initialize LoRA parameters
        # Use smaller initialization for better stability
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)  # Start with zero adaptation
    
    def forward(self, x):
        # Regular linear transformation with frozen weights
        result = F.linear(x, self.weight, self.bias)
        
        # Add LoRA adaptation: x @ A^T @ B^T
        if self.r > 0:
            x_dropout = self.dropout(x)
            lora_output = (x_dropout @ self.lora_A.T) @ self.lora_B.T * self.scaling
            result = result + lora_output
        
        return result
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, r={self.r}'


def create_model(config):
    """Factory function to create FedSA-FTL model with VGG16 backbone"""
    return FedSAFTLModel(
        num_classes=config.get('num_classes', 10),
        model_name=config.get('model_name', 'vgg16'),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.1),
        freeze_backbone=config.get('freeze_backbone', True)
    )