#!/usr/bin/env python3
"""
Utilities for reconstructing LoRA-A matrices from noisy updates.

Supports simple aggregation-based attacks:
  - Average attack (mean of observed noisy updates)
  - SVD attack (dominant singular vector reconstruction)
  - Gradient refinement (DLG-style) as an optional post-processing step
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

TensorDict = Dict[str, torch.Tensor]


@dataclass
class AReconstructionResult:
    reconstructed_A: TensorDict
    method: str
    rounds_used: int


def _prepare_delta_stack(delta_list: List[TensorDict], layer_name: str) -> torch.Tensor:
    """Stack flattened updates for a specific layer."""
    vectors = []
    for delta in delta_list:
        tensor = delta[layer_name]
        vectors.append(tensor.reshape(-1))
    return torch.stack(vectors)


def _average_attack(delta_stack: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    vec = delta_stack.mean(dim=0)
    return vec.view(original_shape)


def _svd_attack(delta_stack: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    delta_stack = delta_stack.to(torch.float64)
    covariance = delta_stack.T @ delta_stack / delta_stack.size(0)
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    top_vec = eigvecs[:, -1]
    mean_vec = delta_stack.mean(dim=0)
    scale = torch.dot(mean_vec, top_vec)
    recon = scale * top_vec
    return recon.to(torch.float32).view(original_shape)


def _gradient_refinement(
    initial: torch.Tensor,
    delta_stack: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
) -> torch.Tensor:
    candidate = initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        expanded = candidate.view(1, -1).expand_as(delta_stack)
        loss = F.mse_loss(expanded, delta_stack)
        loss.backward()
        optimizer.step()
    return candidate.detach().view_as(initial)


def reconstruct_A_from_noisy_updates(
    delta_list: List[TensorDict],
    method: str = "svd",
    attack_config: Optional[Dict] = None,
) -> TensorDict:
    if not delta_list:
        raise ValueError("delta_list must contain at least one update.")

    attack_config = attack_config or {}
    rounds_used = int(attack_config.get("rounds_used", len(delta_list)))
    rounds_used = max(1, min(rounds_used, len(delta_list)))
    selected = delta_list[-rounds_used:]

    method = method.lower()
    refine_steps = int(attack_config.get("refine_steps", 0))
    refine_lr = float(attack_config.get("refine_lr", 0.05))

    layer_names = selected[0].keys()
    reconstructed: TensorDict = {}

    for name in layer_names:
        delta_stack = _prepare_delta_stack(selected, name)
        original_shape = selected[0][name].shape

        if method == "average":
            estimate = _average_attack(delta_stack, original_shape)
        elif method == "svd":
            estimate = _svd_attack(delta_stack, original_shape)
        elif method == "dlg":
            avg = _average_attack(delta_stack, original_shape)
            estimate = _gradient_refinement(avg.reshape(-1), delta_stack, steps=refine_steps or 200, lr=refine_lr)
        else:
            raise ValueError(f"Unknown A reconstruction method: {method}")

        if refine_steps and method != "dlg":
            estimate = _gradient_refinement(estimate.reshape(-1), delta_stack, steps=refine_steps, lr=refine_lr)

        reconstructed[name] = estimate.to(selected[0][name].dtype)

    return reconstructed


__all__ = [
    "AReconstructionResult",
    "reconstruct_A_from_noisy_updates",
]
