#!/usr/bin/env python3
"""
Metrics for evaluating reconstructed LoRA-A matrices.
"""

from __future__ import annotations

from typing import Dict, List

import torch

TensorDict = Dict[str, torch.Tensor]


def _flatten_dict(tensor_dict: TensorDict) -> torch.Tensor:
    parts = []
    for name in sorted(tensor_dict.keys()):
        parts.append(tensor_dict[name].reshape(-1))
    return torch.cat(parts)


def _nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.linalg.norm(pred - target) ** 2
    denominator = torch.linalg.norm(target) ** 2 + 1e-12
    return float((numerator / denominator).item())


def _cosine(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_norm = pred / (pred.norm() + 1e-12)
    target_norm = target / (target.norm() + 1e-12)
    return float(torch.dot(pred_norm, target_norm).item())


def _match_matrix_shapes(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.dim() == 1:
        pred = pred.unsqueeze(1)
    if target.dim() == 1:
        target = target.unsqueeze(1)

    if pred.shape[0] != target.shape[0]:
        rows = min(pred.shape[0], target.shape[0])
        pred = pred[:rows]
        target = target[:rows]

    if pred.shape[1] != target.shape[1]:
        cols = min(pred.shape[1], target.shape[1])
        pred = pred[:, :cols]
        target = target[:, :cols]

    return pred, target


def _orthogonal_procrustes(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _match_matrix_shapes(pred, target)
    if pred.numel() == 0 or target.numel() == 0:
        return torch.eye(1, device=pred.device, dtype=pred.dtype)
    M = target.transpose(0, 1) @ pred
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    Q = U @ Vh
    return Q


def _principal_angles(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    pred, target = _match_matrix_shapes(pred, target)
    # pred, target: (m, r)
    Qp, _ = torch.linalg.qr(pred, mode="reduced")
    Qt, _ = torch.linalg.qr(target, mode="reduced")
    sigma = torch.linalg.svd(Qp.T @ Qt, full_matrices=False).S
    sigma = torch.clamp(sigma, -1.0, 1.0)
    angles = torch.rad2deg(torch.acos(sigma))
    return [float(angle.item()) for angle in angles]


def evaluate_A_similarity(recon: TensorDict, true: TensorDict) -> Dict[str, float]:
    recon_vec = _flatten_dict(recon).to(torch.float64)
    true_vec = _flatten_dict(true).to(torch.float64)

    nmse_raw = _nmse(recon_vec, true_vec)
    cos_raw = _cosine(recon_vec, true_vec)

    nmse_aligned_sum = 0.0
    cos_aligned_sum = 0.0
    nmse_all = 0.0
    denom_all = 0.0
    angles_accum: List[float] = []
    layers_evaluated = 0

    for name in true:
        if name not in recon:
            continue
        pred_layer = recon[name].to(torch.float64)
        true_layer = true[name].to(torch.float64)
        pred_layer, true_layer = _match_matrix_shapes(pred_layer, true_layer)
        if pred_layer.numel() == 0 or true_layer.numel() == 0:
            continue
        nmse_all += torch.linalg.norm(pred_layer - true_layer) ** 2
        denom_all += torch.linalg.norm(true_layer) ** 2 + 1e-12

        Q = _orthogonal_procrustes(pred_layer, true_layer)
        aligned = pred_layer @ Q

        nmse_aligned_sum += _nmse(aligned.reshape(-1), true_layer.reshape(-1))
        cos_aligned_sum += _cosine(aligned.reshape(-1), true_layer.reshape(-1))
        angles_accum.extend(_principal_angles(pred_layer, true_layer))
        layers_evaluated += 1

    if layers_evaluated == 0:
        return {
            "nmse_raw": nmse_raw,
            "cos_raw": cos_raw,
            "nmse_aligned": float('nan'),
            "cos_aligned": float('nan'),
            "mean_principal_angle_deg": float('nan'),
            "nmse_layer_weighted": float('nan'),
        }

    nmse_aligned = nmse_aligned_sum / layers_evaluated
    cos_aligned = cos_aligned_sum / layers_evaluated
    mean_theta = sum(angles_accum) / max(1, len(angles_accum))
    nmse_global = float((nmse_all / denom_all).item()) if denom_all > 0 else float('nan')

    return {
        "nmse_raw": nmse_raw,
        "cos_raw": cos_raw,
        "nmse_aligned": nmse_aligned,
        "cos_aligned": cos_aligned,
        "mean_principal_angle_deg": mean_theta,
        "nmse_layer_weighted": nmse_global,
    }


__all__ = [
    "evaluate_A_similarity",
]
