"""
FedSA-FTL: Federated Share-A Transfer Learning
Hybrid architecture combining FbFTL and FedSA-LoRA principles
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
import copy
import re


class LoRALayer(nn.Module):
    """
    LoRA adaptation layer for the head component
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scale factor
        self.scaling = alpha / rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x @ (A @ B) * scaling"""
        result = self.dropout(x) @ self.lora_A @ self.lora_B
        return result * self.scaling


class FedSAFTLHead(nn.Module):
    """
    Task-specific head with LoRA adaptation for FedSA-FTL
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dims: List[int] = [512, 256],
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 lora_dropout: float = 0.05,
                 use_dropout: bool = True,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank
        
        # Build the base head layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Final classification layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.base_head = nn.Sequential(*layers)
        
        # LoRA adaptation for the final layer only
        final_layer = layers[-1]
        self.lora_adapter = LoRALayer(
            in_features=prev_dim,
            out_features=output_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
        
        # Freeze base parameters
        for param in self.base_head.parameters():
            param.requires_grad = False
            
        # Only LoRA parameters are trainable
        self.lora_adapter.lora_A.requires_grad = True
        self.lora_adapter.lora_B.requires_grad = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base head + LoRA adaptation"""
        # Pass through all layers except the last one
        features = x
        for layer in self.base_head[:-1]:
            features = layer(features)
            
        # Apply base final layer + LoRA adaptation
        base_output = self.base_head[-1](features)
        lora_output = self.lora_adapter(features)
        
        return base_output + lora_output
    
    def get_lora_state_dict(self, include_A: bool = True, include_B: bool = True) -> Dict[str, torch.Tensor]:
        """Get LoRA parameters state dict"""
        state_dict = {}
        if include_A:
            state_dict['lora_A'] = self.lora_adapter.lora_A.data.clone()
        if include_B:
            state_dict['lora_B'] = self.lora_adapter.lora_B.data.clone()
        return state_dict
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor], load_A: bool = True, load_B: bool = True):
        """Load LoRA parameters from state dict"""
        if load_A and 'lora_A' in state_dict:
            self.lora_adapter.lora_A.data = state_dict['lora_A'].clone()
        if load_B and 'lora_B' in state_dict:
            self.lora_adapter.lora_B.data = state_dict['lora_B'].clone()


class FedSAFTLModel(nn.Module):
    """
    FedSA-FTL Model: Frozen backbone + LoRA-adapted head
    """
    def __init__(self, 
                 backbone: nn.Module,
                 num_classes: int,
                 backbone_output_dim: int,
                 head_hidden_dims: List[int] = [512, 256],
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 lora_dropout: float = 0.05):
        super().__init__()
        
        # Frozen backbone (feature extractor)
        self.backbone = backbone
        self.freeze_backbone()
        
        # LoRA-adapted head
        self.head = FedSAFTLHead(
            input_dim=backbone_output_dim,
            output_dim=num_classes,
            hidden_dims=head_hidden_dims,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone -> head"""
        with torch.no_grad():
            features = self.backbone(x)
            
        # If backbone returns a tuple (e.g., some Vision Transformers), take the first element
        if isinstance(features, tuple):
            features = features[0]
            
        # Handle classification token for ViT
        if len(features.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            features = features[:, 0]  # Take [CLS] token
            
        return self.head(features)
    
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only trainable (LoRA) parameters"""
        return {name: param for name, param in self.named_parameters() if param.requires_grad}
    
    def get_lora_A_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only LoRA A matrix parameters for sharing"""
        return self.head.get_lora_state_dict(include_A=True, include_B=False)
    
    def get_lora_B_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only LoRA B matrix parameters (client-specific)"""
        return self.head.get_lora_state_dict(include_A=False, include_B=True)
    
    def load_global_lora_A(self, global_A_state: Dict[str, torch.Tensor]):
        """Load global LoRA A parameters"""
        self.head.load_lora_state_dict(global_A_state, load_A=True, load_B=False)
    
    def save_model_state(self, path: str, round_num: int = 0):
        """Save complete model state"""
        state = {
            'round': round_num,
            'backbone_state_dict': self.backbone.state_dict(),
            'head_state_dict': self.head.state_dict(),
            'lora_A': self.get_lora_A_parameters(),
            'lora_B': self.get_lora_B_parameters()
        }
        torch.save(state, path)
    
    def load_model_state(self, path: str):
        """Load complete model state"""
        state = torch.load(path, map_location='cpu')
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.head.load_state_dict(state['head_state_dict'])
        return state['round']


def create_vision_model(model_name: str = 'vit_base', num_classes: int = 10, lora_rank: int = 8) -> FedSAFTLModel:
    """
    Create a vision model for FedSA-FTL
    """
    if model_name == 'vit_base':
        try:
            from transformers import ViTModel, ViTConfig
            
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                num_labels=num_classes
            )
            
            backbone = ViTModel(config)
            backbone_output_dim = config.hidden_size
            
        except ImportError:
            # Fallback to a simple CNN backbone
            print("transformers not available, using simple CNN backbone")
            backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            backbone_output_dim = 256
            
    elif model_name == 'resnet18':
        from torchvision import models
        backbone = models.resnet18(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        backbone_output_dim = 512
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return FedSAFTLModel(
        backbone=backbone,
        num_classes=num_classes,
        backbone_output_dim=backbone_output_dim,
        lora_rank=lora_rank
    )


def create_nlp_model(model_name: str = 'roberta_base', num_classes: int = 3, lora_rank: int = 8) -> FedSAFTLModel:
    """
    Create an NLP model for FedSA-FTL
    """
    try:
        from transformers import RobertaModel, RobertaConfig
        
        if model_name == 'roberta_base':
            config = RobertaConfig.from_pretrained('roberta-base')
            backbone = RobertaModel.from_pretrained('roberta-base')
            backbone_output_dim = config.hidden_size
            
        elif model_name == 'roberta_large':
            config = RobertaConfig.from_pretrained('roberta-large')
            backbone = RobertaModel.from_pretrained('roberta-large')
            backbone_output_dim = config.hidden_size
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
    except ImportError:
        raise ImportError("transformers library is required for NLP models")
    
    return FedSAFTLModel(
        backbone=backbone,
        num_classes=num_classes,
        backbone_output_dim=backbone_output_dim,
        lora_rank=lora_rank
    )
