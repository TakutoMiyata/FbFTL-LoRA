#!/usr/bin/env python3
"""
ResNet models with LoRA for federated learning
Adapted for CIFAR-100 with FedSA-FTL framework
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional, List


class LoRALayer(nn.Module):
    """LoRA adapter layer for ResNet with separate A and B parameters for FedSA-LoRA with DP"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1, seed=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices A and B as separate parameters for FedSA
        # A matrix: shared across clients with DP
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # B matrix: kept local per client
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights with proper seeding for reproducibility
        if seed is not None:
            # Set seed for reproducible initialization
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # LoRA paper recommended initialization:
        # A: random (Kaiming), B: zero (so A@B = 0 initially)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # LoRA forward: x @ A.T @ B.T * scaling
        # x shape: (batch, ..., in_features)
        # Reshape for matrix operations
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # Apply dropout and LoRA transformation
        x_dropout = self.dropout(x_flat)
        # x_dropout @ A.T -> (batch, rank)
        lora_intermediate = torch.matmul(x_dropout, self.lora_A.t())
        # lora_intermediate @ B.T -> (batch, out_features)
        lora_out = torch.matmul(lora_intermediate, self.lora_B.t()) * self.scaling
        
        # Reshape back to original shape (except last dimension)
        new_shape = original_shape[:-1] + (self.out_features,)
        return lora_out.view(new_shape)


class ResNetWithLoRA(nn.Module):
    """ResNet model with LoRA adapters for federated learning"""
    
    def __init__(self, config):
        super().__init__()
        
        # Load configuration
        self.model_name = config.get('model_name', 'resnet50')
        self.num_classes = config.get('num_classes', 100)
        self.pretrained = config.get('pretrained', True)
        self.lora_r = config.get('lora_r', 8)
        self.lora_alpha = config.get('lora_alpha', 16)
        self.lora_dropout = config.get('lora_dropout', 0.1)
        self.seed = config.get('seed', None)  # For reproducible LoRA initialization
        
        # Load base ResNet model
        if self.model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == 'resnet34':
            self.base_model = models.resnet34(pretrained=self.pretrained)
        elif self.model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == 'resnet101':
            self.base_model = models.resnet101(pretrained=self.pretrained)
        elif self.model_name == 'resnet152':
            self.base_model = models.resnet152(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {self.model_name}")
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get the number of features in the final layer
        num_features = self.base_model.fc.in_features
        
        # Replace the final fully connected layer
        self.base_model.fc = nn.Identity()  # Remove original fc layer
        
        # Add new classification head with LoRA
        self.classifier = nn.Linear(num_features, self.num_classes)
        
        # Add LoRA adapters to selected layers
        self.lora_adapters = nn.ModuleDict()
        
        # Add LoRA to the last residual blocks
        # For ResNet50: layer4 has 3 blocks, each with conv1, conv2, conv3
        if self.model_name in ['resnet50', 'resnet101', 'resnet152']:
            # These models use bottleneck blocks
            for block_idx in range(len(self.base_model.layer4)):
                block = self.base_model.layer4[block_idx]
                
                # Add LoRA after conv3 (the 1x1 conv that expands channels)
                out_channels = block.conv3.out_channels
                self.lora_adapters[f'layer4_{block_idx}_conv3'] = LoRALayer(
                    out_channels, out_channels, 
                    rank=self.lora_r,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                    seed=self.seed + block_idx if self.seed is not None else None
                )
        else:
            # ResNet18/34 use basic blocks
            for block_idx in range(len(self.base_model.layer4)):
                block = self.base_model.layer4[block_idx]
                
                # Add LoRA after conv2
                out_channels = block.conv2.out_channels
                self.lora_adapters[f'layer4_{block_idx}_conv2'] = LoRALayer(
                    out_channels, out_channels,
                    rank=self.lora_r,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                    seed=self.seed + block_idx if self.seed is not None else None
                )
        
        # Add LoRA to classifier as well
        self.classifier_lora = LoRALayer(
            num_features, self.num_classes,
            rank=self.lora_r,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            seed=self.seed + 1000 if self.seed is not None else None  # Unique seed for classifier
        )
        
        # Store feature dimension for feature extraction
        self.feature_dim = num_features
        
        # Initialize shared A matrix flag
        self.use_shared_A = True
    
    def forward(self, x, return_features=False):
        """Forward pass with optional feature extraction"""
        
        # Initial layers
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        # Residual layers
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        
        # Layer 4 with LoRA adapters
        for block_idx, block in enumerate(self.base_model.layer4):
            identity = x
            
            if self.model_name in ['resnet50', 'resnet101', 'resnet152']:
                # Bottleneck block
                out = block.conv1(x)
                out = block.bn1(out)
                out = block.relu(out)
                
                out = block.conv2(out)
                out = block.bn2(out)
                out = block.relu(out)
                
                out = block.conv3(out)
                out = block.bn3(out)
                
                # Apply LoRA
                lora_key = f'layer4_{block_idx}_conv3'
                if lora_key in self.lora_adapters:
                    # Reshape for LoRA (batch, channels, height, width) -> (batch, height*width, channels)
                    b, c, h, w = out.shape
                    out_reshaped = out.permute(0, 2, 3, 1).reshape(b * h * w, c)
                    lora_out = self.lora_adapters[lora_key](out_reshaped)
                    lora_out = lora_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
                    out = out + lora_out
            else:
                # Basic block (ResNet18/34)
                out = block.conv1(x)
                out = block.bn1(out)
                out = block.relu(out)
                
                out = block.conv2(out)
                out = block.bn2(out)
                
                # Apply LoRA
                lora_key = f'layer4_{block_idx}_conv2'
                if lora_key in self.lora_adapters:
                    # Reshape for LoRA
                    b, c, h, w = out.shape
                    out_reshaped = out.permute(0, 2, 3, 1).reshape(b * h * w, c)
                    lora_out = self.lora_adapters[lora_key](out_reshaped)
                    lora_out = lora_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
                    out = out + lora_out
            
            # Handle downsample if exists
            if block.downsample is not None:
                identity = block.downsample(identity)
            
            out += identity
            out = block.relu(out)
            x = out
        
        # Global average pooling
        x = self.base_model.avgpool(x)
        features = torch.flatten(x, 1)
        
        if return_features:
            return features
        
        # Classification with LoRA
        logits = self.classifier(features)
        if self.classifier_lora is not None:
            logits = logits + self.classifier_lora(features)
        
        return logits
    
    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all LoRA parameters (A and B matrices)"""
        lora_params = {}
        
        # Get LoRA adapter parameters (now nn.Parameter)
        for name, adapter in self.lora_adapters.items():
            lora_params[f'{name}_A'] = adapter.lora_A
            lora_params[f'{name}_B'] = adapter.lora_B
        
        # Get classifier LoRA parameters
        if self.classifier_lora is not None:
            lora_params['classifier_lora_A'] = self.classifier_lora.lora_A
            lora_params['classifier_lora_B'] = self.classifier_lora.lora_B
        
        # Also include the main classifier weights
        lora_params['classifier_weight'] = self.classifier.weight
        lora_params['classifier_bias'] = self.classifier.bias
        
        return lora_params
    
    def set_lora_parameters(self, lora_params: Dict[str, torch.Tensor]):
        """Set LoRA parameters from dictionary"""
        
        # Set LoRA adapter parameters (now nn.Parameter)
        for name, adapter in self.lora_adapters.items():
            if f'{name}_A' in lora_params:
                adapter.lora_A.data.copy_(lora_params[f'{name}_A'])
            if f'{name}_B' in lora_params:
                adapter.lora_B.data.copy_(lora_params[f'{name}_B'])
        
        # Set classifier LoRA parameters
        if self.classifier_lora is not None:
            if 'classifier_lora_A' in lora_params:
                self.classifier_lora.lora_A.data.copy_(lora_params['classifier_lora_A'])
            if 'classifier_lora_B' in lora_params:
                self.classifier_lora.lora_B.data.copy_(lora_params['classifier_lora_B'])
        
        # Set main classifier weights
        if 'classifier_weight' in lora_params:
            self.classifier.weight.data.copy_(lora_params['classifier_weight'])
        if 'classifier_bias' in lora_params:
            self.classifier.bias.data.copy_(lora_params['classifier_bias'])
    
    def get_A_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only the A matrix parameters (shared in FedSA with DP)"""
        A_params = {}
        
        # Get A matrices from LoRA adapters
        for name, adapter in self.lora_adapters.items():
            A_params[f'{name}_A'] = adapter.lora_A
        
        # Get A matrix from classifier LoRA
        if self.classifier_lora is not None:
            A_params['classifier_lora_A'] = self.classifier_lora.lora_A
        
        return A_params
    
    def get_B_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only the B matrix parameters (personalized in FedSA)"""
        B_params = {}
        
        # Get B matrices from LoRA adapters
        for name, adapter in self.lora_adapters.items():
            B_params[f'{name}_B'] = adapter.lora_B
        
        # Get B matrix from classifier LoRA
        if self.classifier_lora is not None:
            B_params['classifier_lora_B'] = self.classifier_lora.lora_B
        
        # Include classifier weights as part of personalized parameters
        B_params['classifier_weight'] = self.classifier.weight
        B_params['classifier_bias'] = self.classifier.bias
        
        return B_params
    
    def set_A_parameters(self, A_params: Dict[str, torch.Tensor]):
        """Set only the A matrix parameters (shared from server)"""
        for name, adapter in self.lora_adapters.items():
            if f'{name}_A' in A_params:
                adapter.lora_A.data.copy_(A_params[f'{name}_A'])
        
        if self.classifier_lora is not None and 'classifier_lora_A' in A_params:
            self.classifier_lora.lora_A.data.copy_(A_params['classifier_lora_A'])
    
    def set_B_parameters(self, B_params: Dict[str, torch.Tensor]):
        """Set only the B matrix parameters (local to each client)"""
        for name, adapter in self.lora_adapters.items():
            if f'{name}_B' in B_params:
                adapter.lora_B.data.copy_(B_params[f'{name}_B'])
        
        if self.classifier_lora is not None and 'classifier_lora_B' in B_params:
            self.classifier_lora.lora_B.data.copy_(B_params['classifier_lora_B'])
        
        if 'classifier_weight' in B_params:
            self.classifier.weight.data.copy_(B_params['classifier_weight'])
        if 'classifier_bias' in B_params:
            self.classifier.bias.data.copy_(B_params['classifier_bias'])
    
    def extract_features(self, x):
        """Extract features for FbFTL"""
        return self.forward(x, return_features=True)
    
    def get_A_parameter_groups(self) -> List[torch.nn.Parameter]:
        """Get list of A matrix parameters for DP-SGD"""
        A_params = []
        
        # Collect A matrix parameters
        for name, adapter in self.lora_adapters.items():
            A_params.append(adapter.lora_A)
        
        if self.classifier_lora is not None:
            A_params.append(self.classifier_lora.lora_A)
        
        return A_params
    
    def get_B_parameter_groups(self) -> List[torch.nn.Parameter]:
        """Get list of B matrix parameters (local to client)"""
        B_params = []
        
        # Collect B matrix parameters
        for name, adapter in self.lora_adapters.items():
            B_params.append(adapter.lora_B)
        
        if self.classifier_lora is not None:
            B_params.append(self.classifier_lora.lora_B)
        
        # Add classifier parameters
        B_params.extend([self.classifier.weight, self.classifier.bias])
        
        return B_params
    
    def get_parameter_groups_for_optimizer(self) -> List[Dict]:
        """Get parameter groups for optimizer with separate A and B groups"""
        return [
            {'params': self.get_A_parameter_groups(), 'name': 'lora_A'},
            {'params': self.get_B_parameter_groups(), 'name': 'lora_B'}
        ]


def create_model_resnet(config: Dict) -> ResNetWithLoRA:
    """
    Factory function to create ResNet model with LoRA
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        ResNetWithLoRA model instance
    """
    return ResNetWithLoRA(config)


# For backward compatibility
def create_resnet50_lora(num_classes=100, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """Create ResNet50 with LoRA for CIFAR-100"""
    config = {
        'model_name': 'resnet50',
        'num_classes': num_classes,
        'pretrained': True,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout
    }
    return create_model_resnet(config)