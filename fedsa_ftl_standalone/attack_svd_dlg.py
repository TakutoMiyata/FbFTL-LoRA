#!/usr/bin/env python3
"""
SVD-Feature Leakage + DLG Refinement attack utilities.

This module follows the research plan described in REAME.md:
  1. Extract latent features from LoRA-A updates via SVD.
  2. Optionally invert the features back to the input domain.
  3. Run a Deep Leakage from Gradients (DLG) refinement step that
     matches LoRA-A gradients to recover the original data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

_HAS_PIQ = False
_HAS_TORCHMETRICS = False
try:  # pragma: no cover - optional dependency
    import piq  # type: ignore

    _HAS_PIQ = True
except Exception:  # pragma: no cover - fallback silently
    pass

try:  # pragma: no cover - optional dependency
    from torchmetrics.functional import structural_similarity_index_measure  # type: ignore

    _HAS_TORCHMETRICS = True
except Exception:  # pragma: no cover
    structural_similarity_index_measure = None


TensorDict = Dict[str, torch.Tensor]
MaybeTensorDict = Union[TensorDict, Iterable[torch.Tensor], torch.Tensor]


@dataclass
class SVDLeakageOutput:
    """Container for the SVD feature leakage stage."""

    feature_vectors: torch.Tensor
    reconstructed_inputs: Optional[torch.Tensor]
    singular_values: torch.Tensor
    computation_time: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "singular_energy": float(self.singular_values.square().sum().item()),
            "top_singular_value": float(self.singular_values[0].item()),
            "num_features": int(self.feature_vectors.shape[0]),
            "elapsed_seconds": float(self.computation_time),
        }


@dataclass
class DLGAttackResult:
    """Metrics from the DLG refinement step."""

    reconstruction_mse: float
    reconstruction_l2: float
    cosine_similarity: float
    psnr: Optional[float]
    ssim: Optional[float]
    label_accuracy: Optional[float]
    grad_match: float
    steps: int
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, float]:
        result = {
            "reconstruction_mse": float(self.reconstruction_mse),
            "reconstruction_l2": float(self.reconstruction_l2),
            "cosine_similarity": float(self.cosine_similarity),
            "grad_match": float(self.grad_match),
            "optimization_steps": int(self.steps),
            "elapsed_seconds": float(self.elapsed_seconds),
        }
        if self.psnr is not None:
            result["psnr"] = float(self.psnr)
        if self.ssim is not None:
            result["ssim"] = float(self.ssim)
        if self.label_accuracy is not None:
            result["label_accuracy"] = float(self.label_accuracy)
        return result


def _flatten_tensor_sequence(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    flat_parts: List[torch.Tensor] = []
    for tensor in tensors:
        if tensor is None:
            continue
        flat_parts.append(tensor.reshape(-1))
    if not flat_parts:
        raise ValueError("No tensors provided for flattening.")
    return torch.cat(flat_parts)


def _flatten_tensor_dict(tensor_dict: TensorDict, device: Optional[torch.device] = None) -> torch.Tensor:
    ordered_tensors: List[torch.Tensor] = []
    for key in sorted(tensor_dict.keys()):
        tensor = tensor_dict[key]
        if device is not None:
            tensor = tensor.to(device)
        ordered_tensors.append(tensor.reshape(-1))
    return torch.cat(ordered_tensors)


def _compute_psnr(recon: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(recon, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def _compute_ssim(recon: torch.Tensor, target: torch.Tensor) -> float:
    recon = recon.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    if _HAS_PIQ:  # pragma: no branch
        return float(piq.ssim(recon, target, data_range=1.0).item())  # type: ignore[attr-defined]
    if _HAS_TORCHMETRICS and structural_similarity_index_measure is not None:  # pragma: no branch
        return float(structural_similarity_index_measure(recon, target, data_range=1.0).item())

    # Fallback: luminance/contrast approximation
    mu_x = recon.mean()
    mu_y = target.mean()
    sigma_x = recon.var(unbiased=False)
    sigma_y = target.var(unbiased=False)
    sigma_xy = ((recon - mu_x) * (target - mu_y)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / denominator).item())


def _iter_lora_a_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Collect parameters associated with LoRA-A matrices."""
    params: List[nn.Parameter] = []
    if hasattr(model, "get_A_parameter_groups"):
        groups = model.get_A_parameter_groups()
        for group in groups:
            if isinstance(group, dict):
                params.extend([p for p in group.get("params", []) if p.requires_grad])
            elif isinstance(group, (list, tuple)):
                params.extend([p for p in group if p.requires_grad])
            elif isinstance(group, nn.Parameter):
                if group.requires_grad:
                    params.append(group)
    if params:
        return params

    # Fallback: search by name
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if "lora" in lowered and "a" in lowered:
            params.append(param)
    if not params:
        raise ValueError("Unable to find LoRA-A parameters on the provided model.")
    return params


def _infer_rank_from_delta(delta: torch.Tensor) -> int:
    if delta.dim() < 2:
        raise ValueError("LoRA-A update tensor must be at least 2D (out_dim x rank).")
    return delta.shape[-1]


def svd_feature_leakage(
    delta_A: MaybeTensorDict,
    B_matrix: Optional[torch.Tensor] = None,
    decoder: Optional[nn.Module] = None,
    topk: int = 1,
    device: Optional[torch.device] = None,
) -> Tuple[SVDLeakageOutput, torch.Tensor]:
    """
    Perform SVD-based feature leakage on LoRA-A updates.

    Args:
        delta_A: Either a dict of tensors (per layer) or a single tensor representing ΔA.
        B_matrix: Optional LoRA-B matrix to invert the feature representation.
        decoder: Optional learned decoder G(z) -> x.
        topk: Number of singular directions to keep.
        device: Target device for computations.
    Returns:
        (SVDLeakageOutput, approximated_feature_vector)
    """
    start = time.time()
    if isinstance(delta_A, torch.Tensor):
        delta_concat = delta_A.reshape(-1, delta_A.shape[-1])
    elif isinstance(delta_A, dict):
        flattened = [
            tensor.reshape(-1, tensor.shape[-1])
            for tensor in delta_A.values()
        ]
        delta_concat = torch.cat(flattened, dim=0)
    else:
        flattened = [
            tensor.reshape(-1, tensor.shape[-1])
            for tensor in delta_A  # type: ignore[arg-type]
        ]
        delta_concat = torch.cat(flattened, dim=0)

    if device is not None:
        delta_concat = delta_concat.to(device)

    # ΔA ≈ g (Bx)^T → Extract top singular vectors from right singular space.
    U, S, Vh = torch.linalg.svd(delta_concat, full_matrices=False)
    topk = max(1, min(topk, Vh.shape[0]))
    feature_vectors = Vh[:topk, :] * S[:topk].unsqueeze(1)

    reconstructed_inputs: Optional[torch.Tensor] = None
    feature_estimate = feature_vectors[0]

    if B_matrix is not None:
        device = delta_concat.device
        B_matrix = B_matrix.to(device)
        lstsq_result = torch.linalg.lstsq(B_matrix.T, feature_estimate.unsqueeze(-1))
        reconstructed_inputs = lstsq_result.solution.squeeze(-1)
    elif decoder is not None:
        decoder = decoder.to(delta_concat.device)
        decoder.eval()
        with torch.no_grad():
            reconstructed_inputs = decoder(feature_estimate.unsqueeze(0)).squeeze(0)

    elapsed = time.time() - start
    svd_result = SVDLeakageOutput(
        feature_vectors=feature_vectors.detach().cpu(),
        reconstructed_inputs=None if reconstructed_inputs is None else reconstructed_inputs.detach().cpu(),
        singular_values=S[:topk].detach().cpu(),
        computation_time=elapsed,
    )
    return svd_result, feature_estimate.detach()


def dlg_refinement(
    model: nn.Module,
    delta_A: TensorDict,
    *,
    batch_images: Optional[torch.Tensor] = None,
    batch_labels: Optional[torch.Tensor] = None,
    attack_config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[DLGAttackResult, torch.Tensor, Optional[torch.Tensor]]:
    """
    Run DLG refinement initialized (optionally) from SVD reconstruction.

    Args:
        model: LoRA model exposing get_A_parameter_groups().
        delta_A: Target LoRA-A update dictionary (ground-truth gradient).
        batch_images: Ground-truth batch for metric calculation (not used by attack).
        batch_labels: Ground-truth labels (optional).
        attack_config: Dictionary with LR/steps/etc.
        device: CUDA/CPU device.
    Returns:
        (DLGAttackResult, reconstructed_images, reconstructed_labels)
    """
    attack_config = attack_config or {}
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.train()

    attack_lr = float(attack_config.get("attack_lr", 0.1))
    attack_steps = int(attack_config.get("attack_steps", 2000))
    data_reg = float(attack_config.get("data_reg", 1e-4))
    tv_reg = float(attack_config.get("tv_reg", 1e-5))
    grad_weight = float(attack_config.get("grad_weight", 1.0))
    label_weight = float(attack_config.get("label_weight", 1.0))
    noise_std = float(attack_config.get("init_noise_std", 0.1))
    num_classes = attack_config.get("num_classes")
    optimize_labels = bool(attack_config.get("optimize_labels", True))

    if batch_images is None:
        raise ValueError("batch_images must be provided to determine the reconstruction shape.")

    initial_images = attack_config.get("initial_images")
    batch_images = batch_images.to(device)
    if initial_images is not None:
        dummy_data = initial_images.to(device).clone().detach()
    else:
        dummy_data = batch_images.clone().detach()
    dummy_data.add_(torch.randn_like(dummy_data) * noise_std)
    dummy_data = dummy_data.clamp(0, 1).requires_grad_(True)

    if num_classes is None:
        if batch_labels is not None:
            num_classes = int(batch_labels.max().item() + 1)
        else:
            raise ValueError("num_classes must be provided in attack_config when labels are unknown.")

    if optimize_labels:
        dummy_label_logits = torch.randn(batch_images.size(0), num_classes, device=device, requires_grad=True)
        label_parameters = [dummy_label_logits]
    else:
        label_parameters = []
        if batch_labels is None:
            raise ValueError("batch_labels required when optimize_labels=False.")

    attack_params = [dummy_data] + label_parameters
    optimizer = torch.optim.Adam(attack_params, lr=attack_lr)

    target_grad = _flatten_tensor_dict(delta_A, device=device)
    attack_lora_params = _iter_lora_a_parameters(model)

    start = time.time()
    best_result: Optional[DLGAttackResult] = None
    best_images = None
    best_labels = None
    grad_vector = None

    for step in range(1, attack_steps + 1):
        optimizer.zero_grad()
        model.zero_grad()

        logits = model(dummy_data)
        if optimize_labels:
            soft_labels = F.softmax(dummy_label_logits, dim=-1)
            ce_loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
        else:
            gt_labels = batch_labels.to(device)
            ce_loss = F.cross_entropy(logits, gt_labels)

        priors = data_reg * dummy_data.pow(2).mean()
        if tv_reg > 0:
            tv_loss = F.l1_loss(dummy_data[:, :, 1:, :], dummy_data[:, :, :-1, :]) + F.l1_loss(
                dummy_data[:, :, :, 1:], dummy_data[:, :, :, :-1]
            )
            priors = priors + tv_reg * tv_loss

        grads = torch.autograd.grad(
            ce_loss,
            attack_lora_params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )

        grad_list: List[torch.Tensor] = []
        for grad in grads:
            if grad is None:
                continue
            grad_list.append(grad.reshape(-1))

        if not grad_list:
            raise RuntimeError("Model produced no gradients for LoRA-A parameters.")

        grad_vector = torch.cat(grad_list)
        grad_loss = F.mse_loss(grad_vector, target_grad)
        total_loss = grad_weight * grad_loss + ce_loss + priors

        if optimize_labels:
            label_entropy = -(soft_labels * torch.log(soft_labels + 1e-8)).sum(dim=1).mean()
            total_loss = total_loss + label_weight * label_entropy

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            dummy_data.clamp_(0, 1)

        if step % attack_config.get("log_interval", 100) == 0 or step == attack_steps:
            with torch.no_grad():
                recon = dummy_data.detach()
                target = batch_images.detach()
                mse_val = F.mse_loss(recon, target).item()
                l2_val = F.mse_loss(recon, target, reduction="sum").sqrt().item()
                cos_val = F.cosine_similarity(recon.view(1, -1), target.view(1, -1)).item()
                grad_match = F.mse_loss(grad_vector, target_grad).item() if grad_vector is not None else float("nan")
                psnr = _compute_psnr(recon, target)
                ssim = _compute_ssim(recon, target)
                if batch_labels is not None:
                    predicted_labels = logits.detach().softmax(dim=-1).argmax(dim=-1).cpu()
                    label_acc = (predicted_labels == batch_labels.cpu()).float().mean().item()
                else:
                    label_acc = None

                result = DLGAttackResult(
                    reconstruction_mse=mse_val,
                    reconstruction_l2=l2_val,
                    cosine_similarity=cos_val,
                    psnr=psnr,
                    ssim=ssim,
                    label_accuracy=label_acc,
                    grad_match=grad_match,
                    steps=step,
                    elapsed_seconds=time.time() - start,
                )
                if best_result is None or mse_val < best_result.reconstruction_mse:
                    best_result = result
                    best_images = recon.detach().cpu()
                    best_labels = logits.detach().cpu().softmax(dim=-1).argmax(dim=-1)

    if best_result is None:
        raise RuntimeError("DLG attack failed to produce any intermediate result.")

    return best_result, best_images, best_labels


__all__ = [
    "SVDLeakageOutput",
    "DLGAttackResult",
    "svd_feature_leakage",
    "dlg_refinement",
]
