"""
FedSA-FTL Model Implementation
Combines frozen backbone from FbFTL with LoRA adaptation for the head
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from peft import LoraConfig, get_peft_model, TaskType
import copy


class FedSAFTLModel(nn.Module):
    """
    FedSA-FTL Model with frozen backbone and LoRA-adapted head
    """
    
    def __init__(self, num_classes=10, model_name='google/vit-base-patch16-224-in21k', 
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, freeze_backbone=True):
        super().__init__()
        
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(model_name)
        hidden_size = self.vit.config.hidden_size
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Store LoRA configuration
        self.lora_config = {
            'r': lora_r,
            'alpha': lora_alpha,
            'dropout': lora_dropout
        }
        
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
    
    def forward(self, pixel_values):
        # Extract features with frozen backbone
        with torch.no_grad() if not self.vit.training else torch.enable_grad():
            outputs = self.vit(pixel_values=pixel_values)
            pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        
        # Pass through LoRA-adapted classifier
        logits = self.classifier(pooled_output)
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
                    lora_params[f"{name}.lora_A"] = module.lora_A.data
                if matrix_type in ['B', 'both']:
                    lora_params[f"{name}.lora_B"] = module.lora_B.data
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
                    module.lora_A.data = lora_params[f"{name}.lora_A"]
                if matrix_type in ['B', 'both'] and f"{name}.lora_B" in lora_params:
                    module.lora_B.data = lora_params[f"{name}.lora_B"]


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
        nn.init.kaiming_uniform_(self.lora_A, a=torch.sqrt(torch.tensor(5.0)))
    
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
    """Factory function to create FedSA-FTL model"""
    return FedSAFTLModel(
        num_classes=config.get('num_classes', 10),
        model_name=config.get('model_name', 'google/vit-base-patch16-224-in21k'),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.1),
        freeze_backbone=config.get('freeze_backbone', True)
    )