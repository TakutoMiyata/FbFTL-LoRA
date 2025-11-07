"""
Compact convolutional backbone with LoRA adapters for Dual-B experiments.
Provides helper methods to access/set LoRA A/B matrices and server-injected B matrices.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer with server-controlled B matrix."""

    def __init__(self, in_features, out_features, r=4, alpha=16, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("lora_B_server", torch.zeros(out_features, r))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        if self.r <= 0:
            return base

        if self.dropout.p > 0 and self.training:
            x = self.dropout(x)

        effective_B = self.lora_B + self.lora_B_server
        lora = (x @ self.lora_A.t()) @ effective_B.t() * self.scaling
        return base + lora

    def set_server_B(self, tensor: torch.Tensor):
        if tensor is None:
            self.lora_B_server.zero_()
        else:
            self.lora_B_server.data.copy_(tensor.to(self.lora_B_server.device, self.lora_B_server.dtype))

    def get_server_B(self):
        return self.lora_B_server.data.clone()

    def reset_server_B(self):
        self.lora_B_server.zero_()


class SmallConvNetLoRA(nn.Module):
    """Simple CNN backbone (input 64x64) with a LoRA linear head."""

    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        input_size=64,
        hidden_channels=64,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flattened_dim = (hidden_channels * 2) * 4 * 4

        self.head = LoRALinear(
            flattened_dim,
            num_classes,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

    def _forward_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        feats = self._forward_features(x)
        return self.head(feats)

    def get_lora_params(self, matrix_type="both") -> Dict[str, torch.Tensor]:
        params = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                if matrix_type in ("A", "both"):
                    params[f"{name}.lora_A"] = module.lora_A.data.clone()
                if matrix_type in ("B", "both"):
                    params[f"{name}.lora_B"] = module.lora_B.data.clone()
        return params

    def set_lora_params(self, params: Dict[str, torch.Tensor], matrix_type="both"):
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                key_A = f"{name}.lora_A"
                key_B = f"{name}.lora_B"
                if matrix_type in ("A", "both") and key_A in params:
                    module.lora_A.data.copy_(params[key_A].to(module.lora_A.device, module.lora_A.dtype))
                if matrix_type in ("B", "both") and key_B in params:
                    module.lora_B.data.copy_(params[key_B].to(module.lora_B.device, module.lora_B.dtype))

    def get_server_B_params(self):
        params = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                params[f"{name}.lora_B_server"] = module.get_server_B()
        return params

    def set_server_B_params(self, params: Dict[str, torch.Tensor]):
        if not params:
            return
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                key = f"{name}.lora_B_server"
                if key in params:
                    module.set_server_B(params[key])

    def reset_server_B_params(self):
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.reset_server_B()
