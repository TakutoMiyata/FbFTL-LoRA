#!/usr/bin/env python3
"""
Privacy attack utilities for evaluating differential privacy defenses.

This module implements:
  - Membership Inference Attack (loss-threshold based)
  - Gradient Leakage Attack (DLG-style gradient inversion on LoRA A matrices)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class MembershipAttackResult:
    accuracy: float
    threshold: float
    tpr: float
    fpr: float
    auc: float
    member_loss_mean: float
    nonmember_loss_mean: float
    member_loss_std: float
    nonmember_loss_std: float
    num_members: int
    num_nonmembers: int

    def to_dict(self) -> Dict[str, float]:
        return {
            'attack_accuracy': float(self.accuracy),
            'best_threshold': float(self.threshold),
            'tpr_at_threshold': float(self.tpr),
            'fpr_at_threshold': float(self.fpr),
            'auc': float(self.auc),
            'member_loss_mean': float(self.member_loss_mean),
            'nonmember_loss_mean': float(self.nonmember_loss_mean),
            'member_loss_std': float(self.member_loss_std),
            'nonmember_loss_std': float(self.nonmember_loss_std),
            'num_members': int(self.num_members),
            'num_nonmembers': int(self.num_nonmembers),
        }


@dataclass
class GradientLeakageResult:
    reconstruction_mse: float
    reconstruction_l2: float
    cosine_similarity: float
    label_accuracy: float
    final_grad_match: float
    steps: int
    elapsed_seconds: float
    attacked_layers: Sequence[str]

    def to_dict(self) -> Dict[str, float]:
        return {
            'reconstruction_mse': float(self.reconstruction_mse),
            'reconstruction_l2': float(self.reconstruction_l2),
            'cosine_similarity': float(self.cosine_similarity),
            'label_accuracy': float(self.label_accuracy),
            'final_grad_match': float(self.final_grad_match),
            'optimization_steps': int(self.steps),
            'elapsed_seconds': float(self.elapsed_seconds),
            'attacked_layers': list(self.attacked_layers),
        }


def _ensure_tensor_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Convert labels to integer class indices if they are provided as one-hot vectors.
    """
    if labels.dim() == 2:
        return labels.argmax(dim=1)
    return labels


def _collect_losses(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute per-sample cross-entropy losses for a dataloader.
    """
    was_training = model.training
    model.eval()

    losses: List[torch.Tensor] = []
    batches_processed = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(data)

            if target.dim() > 1:
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(target * log_probs).sum(dim=1)
            else:
                loss = F.cross_entropy(logits, target, reduction='none')

            losses.append(loss.detach().cpu())

            batches_processed += 1
            if max_batches is not None and batches_processed >= max_batches:
                break

    if was_training:
        model.train()

    if not losses:
        return torch.empty(0)

    return torch.cat(losses)


def _compute_auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    if member_scores.size == 0 or nonmember_scores.size == 0:
        return float('nan')

    member_scores = member_scores[:, None]
    nonmember_scores = nonmember_scores[None, :]

    greater = (member_scores > nonmember_scores).mean()
    equal = (member_scores == nonmember_scores).mean()
    return float(greater + 0.5 * equal)


def _search_best_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Find threshold that maximizes attack accuracy when predicting member if score >= threshold.
    Returns (best_threshold, best_accuracy, tpr, fpr).
    """
    assert scores.shape == labels.shape
    order = np.argsort(-scores)  # descending
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    total_pos = labels.sum()
    total_neg = labels.shape[0] - total_pos
    if total_pos == 0 or total_neg == 0:
        baseline_acc = max(total_pos, total_neg) / labels.shape[0]
        return float(scores_sorted[0]), baseline_acc, 0.0, 0.0

    tp_cumsum = np.cumsum(labels_sorted)
    fp_cumsum = np.cumsum(1 - labels_sorted)

    tn = total_neg - fp_cumsum
    accuracy = (tp_cumsum + tn) / labels.shape[0]

    best_idx = int(np.argmax(accuracy))
    best_threshold = scores_sorted[best_idx]
    best_accuracy = float(accuracy[best_idx])

    tp = tp_cumsum[best_idx]
    fp = fp_cumsum[best_idx]

    tpr = float(tp / total_pos)
    fpr = float(fp / total_neg)

    return best_threshold, best_accuracy, tpr, fpr


def run_membership_inference_attack(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: torch.device,
    max_member_batches: Optional[int] = None,
    max_nonmember_batches: Optional[int] = None,
) -> MembershipAttackResult:
    """
    Perform a loss-threshold membership inference attack.
    """
    member_losses = _collect_losses(model, member_loader, device, max_member_batches)
    nonmember_losses = _collect_losses(model, nonmember_loader, device, max_nonmember_batches)

    if member_losses.numel() == 0 or nonmember_losses.numel() == 0:
        raise ValueError("Cannot run membership inference attack with zero samples.")

    member_scores = -member_losses.numpy()
    nonmember_scores = -nonmember_losses.numpy()

    all_scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones_like(member_scores),
        np.zeros_like(nonmember_scores),
    ])

    threshold, accuracy, tpr, fpr = _search_best_threshold(all_scores, labels)
    auc = _compute_auc(member_scores, nonmember_scores)

    return MembershipAttackResult(
        accuracy=accuracy,
        threshold=threshold,
        tpr=tpr,
        fpr=fpr,
        auc=auc,
        member_loss_mean=float(member_losses.mean().item()),
        nonmember_loss_mean=float(nonmember_losses.mean().item()),
        member_loss_std=float(member_losses.std(unbiased=False).item()),
        nonmember_loss_std=float(nonmember_losses.std(unbiased=False).item()),
        num_members=int(member_losses.numel()),
        num_nonmembers=int(nonmember_losses.numel()),
    )


def _prepare_attack_parameters(
    model: nn.Module,
    initial_params: Dict[str, torch.Tensor],
    delta_params: Dict[str, torch.Tensor],
    target_layer_names: Sequence[str],
    device: torch.device,
    learning_rate: float,
) -> Tuple[List[str], List[torch.nn.Parameter], List[torch.Tensor]]:
    """
    Restore model parameters to pre-update values and compute target gradients.
    Returns attacked layer names, parameter references, and target gradients.
    """
    attacked_layers: List[str] = []
    param_refs: List[torch.nn.Parameter] = []
    target_grads: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if name in initial_params:
            param.data.copy_(initial_params[name].to(device=device, dtype=param.data.dtype))

    for name in target_layer_names:
        if name not in delta_params:
            continue

        for param_name, param in model.named_parameters():
            if param_name == name:
                attacked_layers.append(name)
                param_refs.append(param)
                grad = -delta_params[name].to(device=device, dtype=param.dtype) / max(learning_rate, 1e-12)
                target_grads.append(grad)
                break

    if not attacked_layers:
        raise ValueError("No valid target layers found for gradient leakage attack.")

    return attacked_layers, param_refs, target_grads


def run_gradient_leakage_attack(
    model: nn.Module,
    attack_payload: Dict,
    device: torch.device,
    attack_config: Dict,
) -> GradientLeakageResult:
    """
    Perform a gradient inversion attack on sanitized LoRA A updates.

    Args:
        model: Trained model (will be deep-copied internally)
        attack_payload: Dictionary with keys:
            - 'initial_A_params': dict(str -> tensor)
            - 'delta_A_params': dict(str -> tensor)
            - 'batch_images': tensor [B,C,H,W]
            - 'batch_labels': tensor [B] or [B, num_classes]
            - 'lr': float (learning rate used for A optimizer)
            - 'num_classes': int (optional)
        attack_config: Dictionary controlling the optimization process
        device: Target device for attack
    """
    batch_images: torch.Tensor = attack_payload['batch_images']
    batch_labels: torch.Tensor = attack_payload['batch_labels']

    lr = float(attack_payload.get('lr', 0.001))
    max_steps = int(attack_config.get('optimization_steps', 200))
    attack_lr = float(attack_config.get('attack_lr', 0.1))
    max_layers = int(attack_config.get('max_layers', 2))
    target_layer_names = attack_config.get('target_layers')
    optimize_labels = bool(attack_config.get('optimize_labels', True))
    l2_reg = float(attack_config.get('l2_regularizer', 1e-4))
    max_time = float(attack_config.get('max_seconds', 10.0))

    model_clone = copy.deepcopy(model).to(device)
    model_clone.eval()

    initial_params: Dict[str, torch.Tensor] = attack_payload['initial_A_params']
    delta_params: Dict[str, torch.Tensor] = attack_payload['delta_A_params']

    if target_layer_names:
        target_layers = list(target_layer_names)
    else:
        target_layers = list(delta_params.keys())[:max_layers]

    attacked_layers, param_refs, target_grads = _prepare_attack_parameters(
        model_clone,
        initial_params,
        delta_params,
        target_layers,
        device,
        lr,
    )

    batch_images = batch_images.to(device)
    batch_labels = batch_labels.to(device)
    true_labels = _ensure_tensor_labels(batch_labels).detach()

    dummy_data = torch.randn_like(batch_images, device=device, requires_grad=True)

    with torch.no_grad():
        test_output = model_clone(batch_images)
    num_classes = attack_payload.get('num_classes', int(test_output.shape[1]))

    optim_params: List[torch.Tensor] = [dummy_data]
    if optimize_labels:
        dummy_label_logits = torch.randn(
            (batch_images.shape[0], num_classes),
            device=device,
            requires_grad=True,
        )
        optim_params.append(dummy_label_logits)
    else:
        dummy_label_logits = None

    optimizer = torch.optim.Adam(optim_params, lr=attack_lr)

    start_time = time.perf_counter()
    final_grad_match = float('nan')

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        model_clone.zero_grad(set_to_none=True)

        pred = model_clone(dummy_data)

        if dummy_label_logits is not None:
            label_distribution = F.softmax(dummy_label_logits, dim=-1)
            loss = -(label_distribution * F.log_softmax(pred, dim=-1)).sum(dim=1).mean()
        else:
            labels_int = _ensure_tensor_labels(batch_labels)
            loss = F.cross_entropy(pred, labels_int)

        grads = torch.autograd.grad(
            loss,
            param_refs,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )

        grad_match = torch.zeros(1, device=device)
        for g_est, g_target in zip(grads, target_grads):
            grad_match = grad_match + F.mse_loss(g_est, g_target)

        if l2_reg > 0:
            grad_match = grad_match + l2_reg * (dummy_data ** 2).mean()

        grad_match.backward()
        optimizer.step()

        final_grad_match = float(grad_match.detach().item())

        if (time.perf_counter() - start_time) > max_time:
            break

    with torch.no_grad():
        reconstruction_mse = F.mse_loss(dummy_data, batch_images).item()
        reconstruction_l2 = torch.norm(dummy_data - batch_images).item()
        cosine_similarity = F.cosine_similarity(
            dummy_data.flatten(),
            batch_images.flatten(),
            dim=0,
        ).item()

        if dummy_label_logits is not None:
            recovered_labels = torch.softmax(dummy_label_logits, dim=-1).argmax(dim=1)
        else:
            recovered_labels = _ensure_tensor_labels(batch_labels)

        label_accuracy = (recovered_labels == true_labels).float().mean().item()

    elapsed = time.perf_counter() - start_time

    return GradientLeakageResult(
        reconstruction_mse=reconstruction_mse,
        reconstruction_l2=reconstruction_l2,
        cosine_similarity=cosine_similarity,
        label_accuracy=label_accuracy,
        final_grad_match=final_grad_match,
        steps=step,
        elapsed_seconds=elapsed,
        attacked_layers=attacked_layers,
    )

