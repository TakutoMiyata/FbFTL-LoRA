"""
FedSA-FTL Vision Transformer (ViT) Model Implementation
ViT backbone with LoRA adaptation following FbFTL paper approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Tuple, Optional


class VisionTransformer(nn.Module):
    """
    Vision Transformer implementation for CIFAR-100
    Following ViT-Base/16 architecture with modifications for 32x32 images
    """
    
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100, 
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2  # 64 patches for 32x32/4x4
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate
            ) for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize classification head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Check input dimensions
        assert H == self.img_size and W == self.img_size, \
            f"Input image size mismatch: got ({H}, {W}), expected ({self.img_size}, {self.img_size})"
        
        # Patch embedding: (B, 3, 32, 32) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        # Ensure correct reshape: the Conv2d outputs (B, embed_dim, H/P, W/P)
        # We need to reshape to (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Verify patch count
        assert x.shape[1] == self.num_patches, f"Patch count mismatch: got {x.shape[1]}, expected {self.num_patches}"
        
        # Add class token: (B, num_patches, embed_dim) -> (B, num_patches+1, embed_dim)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Classification (use CLS token)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP"""
    
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, 
                 drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, drop_rate)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, embed_dim=384, num_heads=6, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation"""
    
    def __init__(self, in_features, hidden_features, drop_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_rate)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FedSAFTLModelViT(nn.Module):
    """
    FedSA-FTL Model with Vision Transformer backbone and LoRA adaptation
    Following FbFTL paper approach with ViT for CIFAR-100
    """
    
    def __init__(self, num_classes=100, model_name='vit_small', 
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, freeze_backbone=True):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes
        self.opacus_mode = False  # Flag for Opacus compatibility
        
        # Store LoRA configuration
        self.lora_config = {
            'r': lora_r,
            'alpha': lora_alpha,
            'dropout': lora_dropout
        }
        
        # Initialize ViT backbone
        self._init_vit_backbone(model_name, num_classes)
        
        # Apply LoRA to attention and MLP layers
        self._apply_lora_to_transformer()
    
    def _init_vit_backbone(self, model_name, num_classes):
        """Initialize Vision Transformer backbone"""
        print(f"Creating Vision Transformer model: {model_name}")
        
        if model_name == 'vit_tiny':
            self.backbone = VisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4.0,
                num_classes=num_classes
            )
        elif model_name == 'vit_small':
            self.backbone = VisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4.0,
                num_classes=num_classes
            )
        elif model_name == 'vit_base':
            self.backbone = VisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported ViT variant: {model_name}")
        
        # Freeze backbone if specified (following FbFTL approach)
        if self.freeze_backbone:
            print("Freezing ViT backbone parameters...")
            for name, param in self.backbone.named_parameters():
                if 'head' not in name:  # Don't freeze the classification head
                    param.requires_grad = False
    
    def _apply_lora_to_transformer(self):
        """Apply LoRA to transformer attention and MLP layers"""
        for block_idx, block in enumerate(self.backbone.blocks):
            # Apply LoRA to attention QKV projection
            original_qkv = block.attn.qkv
            block.attn.qkv = LoRALinear(
                original_qkv.in_features,
                original_qkv.out_features,
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['alpha'],
                lora_dropout=self.lora_config['dropout'],
                original_weight=original_qkv.weight.data.clone(),
                original_bias=original_qkv.bias.data.clone() if original_qkv.bias is not None else None
            )
            
            # Apply LoRA to attention output projection
            original_proj = block.attn.proj
            block.attn.proj = LoRALinear(
                original_proj.in_features,
                original_proj.out_features,
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['alpha'],
                lora_dropout=self.lora_config['dropout'],
                original_weight=original_proj.weight.data.clone(),
                original_bias=original_proj.bias.data.clone() if original_proj.bias is not None else None
            )
            
            # Apply LoRA to MLP layers
            original_fc1 = block.mlp.fc1
            block.mlp.fc1 = LoRALinear(
                original_fc1.in_features,
                original_fc1.out_features,
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['alpha'],
                lora_dropout=self.lora_config['dropout'],
                original_weight=original_fc1.weight.data.clone(),
                original_bias=original_fc1.bias.data.clone() if original_fc1.bias is not None else None
            )
            
            original_fc2 = block.mlp.fc2
            block.mlp.fc2 = LoRALinear(
                original_fc2.in_features,
                original_fc2.out_features,
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['alpha'],
                lora_dropout=self.lora_config['dropout'],
                original_weight=original_fc2.weight.data.clone(),
                original_bias=original_fc2.bias.data.clone() if original_fc2.bias is not None else None
            )
        
        # Apply LoRA to classification head
        original_head = self.backbone.head
        self.backbone.head = LoRALinear(
            original_head.in_features,
            original_head.out_features,
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['alpha'],
            lora_dropout=self.lora_config['dropout'],
            original_weight=original_head.weight.data.clone(),
            original_bias=original_head.bias.data.clone() if original_head.bias is not None else None
        )
    
    def set_opacus_mode(self, enabled=True):
        """
        Enable/disable Opacus mode to handle dropout compatibility
        When enabled, replaces all dropout layers with Identity (no-op)
        """
        self.opacus_mode = enabled
        
        if enabled:
            # Replace all dropout layers with Identity for Opacus compatibility
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    # Store original dropout rate for potential restoration
                    if not hasattr(module, '_original_p'):
                        module._original_p = module.p
                    module.p = 0.0  # Disable dropout
                    module.eval()  # Set to eval mode to disable dropout
        else:
            # Restore original dropout behavior
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    if hasattr(module, '_original_p'):
                        module.p = module._original_p
                        module.train()  # Restore training mode
    
    def forward(self, x):
        """Forward pass for ViT"""
        return self.backbone(x)
    
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
        std = 0.01 / math.sqrt(max(out_features, in_features))
        nn.init.normal_(self.lora_A, mean=0.0, std=std)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Regular linear transformation with frozen weights
        result = F.linear(x, self.weight, self.bias)
        
        # Add LoRA adaptation
        if self.r > 0:
            # Skip dropout if dropout module is in eval mode (Opacus compatibility)
            if hasattr(self, 'dropout') and isinstance(self.dropout, nn.Dropout):
                if self.dropout.training and self.dropout.p > 0:
                    try:
                        x_dropout = self.dropout(x)
                    except RuntimeError as e:
                        if "vmap" in str(e) or "randomness" in str(e):
                            # If in vmap context (Opacus), skip dropout
                            x_dropout = x
                        else:
                            raise e
                else:
                    x_dropout = x
            else:
                x_dropout = x
            
            lora_output = (x_dropout @ self.lora_A.T) @ self.lora_B.T * self.scaling
            result = result + lora_output
        
        return result
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, r={self.r}'


def create_model_vit(config):
    """Factory function to create FedSA-FTL model with ViT backbone"""
    return FedSAFTLModelViT(
        num_classes=config.get('num_classes', 100),
        model_name=config.get('model_name', 'vit_small'),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.1),
        freeze_backbone=config.get('freeze_backbone', True)
    )