#!/usr/bin/env python3
"""
Utility functions for shadow training to assist gradient leakage attacks.

The basic strategy:
  1. Generate synthetic LoRA-A updates (ΔA) using an attacker-controlled dataset.
  2. Train a lightweight decoder that maps flattened ΔA tensors to approximate images.
  3. Use the decoder output as an informed initialization for DLG-style reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:  # pragma: no cover - optional dependency
    import torchvision
    from torchvision import transforms
    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover
    torchvision = None
    transforms = None
    _HAS_TORCHVISION = False

from src.backbones_bit import make_bit_model_with_lora  # type: ignore


def _clone_A_parameters(model) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.get_A_parameters().items()
    }


def flatten_delta_dict(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten LoRA-A delta dictionary into a single vector with deterministic ordering."""
    parts: List[torch.Tensor] = []
    for name in sorted(delta.keys()):
        parts.append(delta[name].reshape(-1))
    return torch.cat(parts)


def _compute_delta_for_batch(
    model,
    data: torch.Tensor,
    labels: torch.Tensor,
    lr: float,
) -> Dict[str, torch.Tensor]:
    """Perform one SGD step on A parameters and return ΔA."""
    pre_state = _clone_A_parameters(model)
    optimizer = torch.optim.SGD(model.get_A_parameter_groups(), lr=lr)
    model.zero_grad()
    optimizer.zero_grad()
    logits = model(data)
    if labels.dim() > 1:
        loss = -(labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    post_state = model.get_A_parameters()
    delta = {
        name: post_state[name].detach().cpu() - pre_state[name].detach().cpu()
        for name in pre_state
    }
    # Restore original parameters to avoid cumulative updates
    model.set_A_parameters(pre_state)
    model.zero_grad()
    optimizer.zero_grad()
    return delta


@dataclass
class ShadowDecoderConfig:
    enabled: bool = False
    dataset: str = "mnist"
    num_samples: int = 128
    batch_size: int = 64
    client_lr: float = 0.05
    decoder_hidden: int = 1024
    decoder_lr: float = 1e-3
    decoder_epochs: int = 5
    cache_key: Optional[str] = None


class ShadowDecoder(nn.Module):
    def __init__(self, input_dim: int, image_shape: Sequence[int], hidden_dim: int = 1024):
        super().__init__()
        output_dim = int(math.prod(image_shape))
        self.image_shape = tuple(image_shape)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.net(x)
        return flat.view(-1, *self.image_shape)


_SHADOW_DECODER_CACHE: Dict[str, ShadowDecoder] = {}


def _build_shadow_transform(dataset_name: str, input_size: int) -> transforms.Compose:
    if not _HAS_TORCHVISION:
        raise RuntimeError("torchvision is required for shadow training.")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    transform_list = [transforms.Resize((input_size, input_size))]
    if dataset_name.lower() in {"mnist", "fashionmnist"}:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return transforms.Compose(transform_list)


def _load_shadow_dataset(dataset_name: str, input_size: int, data_dir: str, batch_size: int) -> DataLoader:
    if not _HAS_TORCHVISION:
        raise RuntimeError("torchvision is required for shadow training.")

    transform = _build_shadow_transform(dataset_name, input_size)
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root=data_dir, split="train", download=True, transform=transform)
    elif dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported shadow dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def _collect_shadow_pairs(
    model,
    dataloader: DataLoader,
    num_samples: int,
    client_lr: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect (flattened ΔA, image) pairs from attacker-controlled data."""
    delta_vectors: List[torch.Tensor] = []
    image_tensors: List[torch.Tensor] = []
    model.train()

    for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)
        delta = _compute_delta_for_batch(model, data, labels, lr=client_lr)
        delta_vec = flatten_delta_dict(delta)
        delta_vectors.append(delta_vec)
        image_tensors.append(data.detach().cpu())
        if len(delta_vectors) >= num_samples:
            break

    if not delta_vectors:
        raise RuntimeError("Failed to collect any shadow samples.")

    deltas = torch.stack(delta_vectors)
    images = torch.stack(image_tensors)
    return deltas, images


def _train_decoder_from_samples(
    delta_matrix: torch.Tensor,
    images: torch.Tensor,
    cfg: ShadowDecoderConfig,
    device: torch.device,
) -> ShadowDecoder:
    input_dim = delta_matrix.shape[1]
    if images.dim() == 5:
        image_shape = images.shape[1:]
    elif images.dim() == 4:
        image_shape = images.shape[1:]
    else:
        raise ValueError(f"Unexpected target tensor shape: {images.shape}")
    decoder = ShadowDecoder(input_dim, image_shape, hidden_dim=cfg.decoder_hidden).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder_lr)

    dataset = TensorDataset(delta_matrix, images.view(images.shape[0], -1))
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)), shuffle=True)

    decoder.train()
    for epoch in range(cfg.decoder_epochs):
        for batch_delta, batch_target in loader:
            batch_delta = batch_delta.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            pred = decoder(batch_delta).view_as(batch_target)
            loss = F.mse_loss(pred, batch_target)
            loss.backward()
            optimizer.step()

    return decoder


def get_shadow_decoder(
    config: Dict,
    device: torch.device,
    shadow_cfg: Dict,
    global_A_params: Optional[Dict[str, torch.Tensor]] = None,
) -> ShadowDecoder:
    """Train (or fetch cached) shadow decoder for the current configuration."""
    merged_cfg = ShadowDecoderConfig(
        enabled=shadow_cfg.get("enabled", False),
        dataset=shadow_cfg.get("dataset", config.get("data", {}).get("dataset_name", "mnist")),
        num_samples=int(shadow_cfg.get("num_samples", 128)),
        batch_size=int(shadow_cfg.get("batch_size", config.get("data", {}).get("batch_size", 64))),
        client_lr=float(shadow_cfg.get("client_lr", config.get("training", {}).get("lr", 0.05))),
        decoder_hidden=int(shadow_cfg.get("decoder_hidden", 1024)),
        decoder_lr=float(shadow_cfg.get("decoder_lr", 1e-3)),
        decoder_epochs=int(shadow_cfg.get("decoder_epochs", 5)),
        cache_key=shadow_cfg.get("cache_key"),
    )

    if not merged_cfg.enabled:
        raise ValueError("Shadow training requested but not enabled in configuration.")

    cache_key = merged_cfg.cache_key or f"{merged_cfg.dataset}_{merged_cfg.num_samples}_{merged_cfg.decoder_hidden}"
    if cache_key in _SHADOW_DECODER_CACHE:
        return _SHADOW_DECODER_CACHE[cache_key].to(device)

    input_size = int(config.get("data", {}).get("input_size", 224))
    data_dir = config.get("data", {}).get("data_dir", "./data")

    shadow_loader = _load_shadow_dataset(
        dataset_name=merged_cfg.dataset,
        input_size=input_size,
        data_dir=data_dir,
        batch_size=merged_cfg.batch_size,
    )

    shadow_model = make_bit_model_with_lora(config).to(device)
    shadow_model.train()
    if global_A_params:
        try:
            shadow_model.set_A_parameters({k: v.to(device) for k, v in global_A_params.items()})
        except AttributeError:
            pass

    deltas, images = _collect_shadow_pairs(
        shadow_model,
        shadow_loader,
        num_samples=merged_cfg.num_samples,
        client_lr=merged_cfg.client_lr,
        device=device,
    )

    decoder = _train_decoder_from_samples(deltas, images, merged_cfg, device)
    _SHADOW_DECODER_CACHE[cache_key] = decoder.cpu()
    return decoder


__all__ = [
    "ShadowDecoder",
    "ShadowDecoderConfig",
    "flatten_delta_dict",
    "get_shadow_decoder",
]
