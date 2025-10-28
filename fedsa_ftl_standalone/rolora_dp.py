#!/usr/bin/env python3
"""
RoLoRA-DP utilities for round-wise orthogonal rotation of LoRA-A updates.

Implements the defense described in REAME.md:
  1. Generate a shared orthogonal matrix per round.
  2. Clients rotate their ΔA updates before communication.
  3. Server aggregates rotated updates and reverts the rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch

TensorDict = Dict[str, torch.Tensor]


def generate_random_orthogonal_matrix(
    seed: int,
    rank: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a deterministic orthogonal matrix Q_t ∈ R^{r×r}.
    """
    if rank <= 0:
        raise ValueError("rank must be positive.")
    generator = torch.Generator(device=device or "cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(rank, rank, generator=generator, device=device or "cpu", dtype=dtype)
    q, r = torch.linalg.qr(gaussian)
    # Enforce right-handedness for stability
    d = torch.diag(r)
    phases = torch.sign(d)
    phases[phases == 0] = 1
    q = q * phases
    return q


def _apply_rotation_to_tensor(tensor: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 2:
        raise ValueError("LoRA-A tensor must be at least 2D (out_dim x rank).")
    if tensor.shape[-1] != Q.shape[0]:
        raise ValueError(f"Rank mismatch between tensor ({tensor.shape[-1]}) and Q ({Q.shape[0]}).")
    orig_device = tensor.device
    rotated = tensor.to(Q.device) @ Q
    return rotated.to(orig_device)


def apply_rolora_dp(delta_A: torch.Tensor | TensorDict, Q: torch.Tensor) -> torch.Tensor | TensorDict:
    """
    Rotate LoRA-A updates prior to communication.
    """
    if isinstance(delta_A, dict):
        return {name: _apply_rotation_to_tensor(tensor, Q) for name, tensor in delta_A.items()}
    return _apply_rotation_to_tensor(delta_A, Q)


def aggregate_with_inverse_rotation(
    rotated_updates: Sequence[torch.Tensor | TensorDict],
    Q: torch.Tensor,
) -> torch.Tensor | TensorDict:
    """
    Aggregate rotated updates and revert back to the original subspace.
    """
    if not rotated_updates:
        raise ValueError("rotated_updates cannot be empty.")

    if isinstance(rotated_updates[0], dict):
        aggregated: Dict[str, torch.Tensor] = {}
        keys = rotated_updates[0].keys()
        for key in keys:
            stacked = torch.stack([update[key] for update in rotated_updates])
            mean_update = stacked.mean(dim=0)
            aggregated[key] = mean_update @ Q.T
        return aggregated

    stacked = torch.stack([tensor for tensor in rotated_updates])
    mean_tensor = stacked.mean(dim=0)
    return mean_tensor @ Q.T


@dataclass
class RoLoRAContext:
    """
    Convenience helper that manages per-round orthogonal rotations.
    """

    base_seed: int
    rank: int
    device: torch.device | None = None
    dtype: torch.dtype = torch.float32

    def matrix_for_round(self, round_idx: int) -> torch.Tensor:
        return generate_random_orthogonal_matrix(
            self.base_seed + round_idx,
            self.rank,
            device=self.device,
            dtype=self.dtype,
        )


__all__ = [
    "generate_random_orthogonal_matrix",
    "apply_rolora_dp",
    "aggregate_with_inverse_rotation",
    "RoLoRAContext",
]
