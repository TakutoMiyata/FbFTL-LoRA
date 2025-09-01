"""
FedSA-FTL Model Implementation
Combines frozen backbone from FbFTL with LoRA adaptation for the head
Uses VGG-16 backbone as in FbFTL paper
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import copy
import math


class FedSAFTLModel(nn.Module):
    """
    FedSA-FTL Model with frozen VGG-16 backbone and LoRA-adapted head
    Following FbFTL paper approach with ImageNet pre-trained VGG-16
    """
    
    def __init__(self, num_classes=100, model_name='vgg16', 
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, freeze_backbone=True):
        super().__init__()
        
        # Load pre-trained VGG-16 model (ImageNet pre-trained like FbFTL paper)
        print(f"Loading pre-trained VGG-16 model from ImageNet")
        if model_name == 'vgg16':
            self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg16_bn':
            self.backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'vgg16' or 'vgg16_bn'")
        
        # Get feature dimension (VGG-16 classifier input: 25088)
        feature_dim = self.backbone.classifier[0].in_features  # 512 * 7 * 7 = 25088
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add adaptive average pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classification head with similar structure to original VGG classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
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
    
    def _apply_lora_to_classifier(self):
        """Apply LoRA decomposition to linear layers in classifier"""
        for i, module in enumerate(self.classifier):
            if isinstance(module, nn.Linear):
                # Replace with LoRA linear layer
                self.classifier[i] = LoRALinear(
                    module.in_features,
                    module.out_features,
                    r=self.lora_config['r'],
                    lora_alpha=self.lora_config['alpha'],
                    lora_dropout=self.lora_config['dropout']
                )
    
    def forward(self, x):
        # Extract features with frozen VGG-16 backbone
        features = self.backbone.features(x)
        
        # Adaptive pooling to ensure consistent feature map size
        features = self.adaptive_pool(features)
        
        # Flatten features
        features = torch.flatten(features, 1)
        
        # Detach to ensure no gradients flow back to frozen backbone if frozen
        if not self.backbone.training:
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
    
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Frozen pre-trained weights (initialized with standard Linear initialization)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.dropout = nn.Dropout(lora_dropout)
        
        # Initialize B to zero for zero initialization of BA
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x):
        # Regular linear transformation with frozen weights
        result = F.linear(x, self.weight, self.bias)
        
        # Add LoRA adaptation
        x_dropout = self.dropout(x)
        lora_output = x_dropout @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return result + lora_output
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, r={self.r}'


import torch.nn.functional as F

def create_model(config):
    """Factory function to create FedSA-FTL model with VGG-16 backbone"""
    return FedSAFTLModel(
        num_classes=config.get('num_classes', 100),
        model_name=config.get('model_name', 'vgg16'),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.1),
        freeze_backbone=config.get('freeze_backbone', True)
    )