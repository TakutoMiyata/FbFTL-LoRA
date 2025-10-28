#!/usr/bin/env python3
"""
Evaluation pipeline for the RoLoRA-DP privacy attack/defense suite.

Usage example:
    python evaluate_privacy_leakage.py \
        --config configs/bit_mnist_svhn.yaml \
        --payload artifacts/example_payload.pt \
        --output results_privacy_attack.csv \
        --apply_rolora \
        --rolora-seed 1234
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from attack_svd_dlg import dlg_refinement, svd_feature_leakage
from rolora_dp import (
    aggregate_with_inverse_rotation,
    apply_rolora_dp,
    generate_random_orthogonal_matrix,
)
from src.backbones_bit import make_bit_model_with_lora

try:  # pragma: no cover - optional dependency
    from torchvision.utils import save_image

    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover
    save_image = None
    _HAS_TORCHVISION = False


def _tensor_from_payload(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return torch.tensor(value)
    if isinstance(value, dict):
        if "__ndarray__" in value:
            data = torch.tensor(value["__ndarray__"])
            return data
        return {k: _tensor_from_payload(v) for k, v in value.items()}
    return torch.tensor(value)


def load_payload(path: Path) -> Dict:
    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
    elif path.suffix == ".json":
        with path.open("r") as f:
            payload = json.load(f)
    else:
        raise ValueError(f"Unsupported payload format: {path.suffix}")
    if not isinstance(payload, dict):
        raise ValueError("Payload file must contain a dictionary.")
    return payload


def ensure_tensor_dict(payload_section) -> Dict[str, torch.Tensor]:
    tensor_dict: Dict[str, torch.Tensor] = {}
    if isinstance(payload_section, dict):
        for key, value in payload_section.items():
            tensor = _tensor_from_payload(value)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Unable to convert payload field '{key}' to tensor.")
            tensor_dict[key] = tensor.float()
    else:
        raise ValueError("Expected dictionary for tensor payload section.")
    return tensor_dict


class SimpleLoRAModel(nn.Module):
    """
    Lightweight fallback model that mimics the LoRA interfaces when BiT is unavailable.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, rank: int = 8):
        super().__init__()
        channels, height, width = input_shape
        hidden_dim = 256
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((height // 2, width // 2)),
            nn.Flatten(),
            nn.Linear(32 * (height // 2) * (width // 2), hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.lora_A = nn.Parameter(torch.randn(hidden_dim, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(rank, num_classes) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        base_logits = self.classifier(feats)
        lora_logits = feats @ self.lora_A @ self.lora_B
        return base_logits + lora_logits

    def get_A_parameter_groups(self):
        return [{"params": [self.lora_A]}]

    def get_B_parameter_groups(self):
        return [{"params": [self.lora_B]}]

    def get_A_parameters(self):
        return {"lora_A": self.lora_A.data.clone()}

    def set_A_parameters(self, params):
        if "lora_A" in params:
            self.lora_A.data.copy_(params["lora_A"].to(self.lora_A.device))


def write_csv_row(csv_path: Path, row: Dict[str, float]):
    fieldnames = [
        "config",
        "dp_epsilon",
        "dp_delta",
        "apply_rolora",
        "rolora_seed",
        "rolora_secret_shared",
        "attack_ssim",
        "attack_psnr",
        "attack_mse",
        "attack_label_acc",
        "gradient_mse",
        "singular_energy",
        "top_singular_value",
        "train_accuracy",
        "test_accuracy",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_reconstruction_artifacts(
    artifact_dir: Optional[Path],
    recon_images: torch.Tensor,
    recon_labels: Optional[torch.Tensor],
    prefix: str,
):
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = artifact_dir / f"{prefix}_reconstruction.pt"
    torch.save(
        {
            "images": recon_images,
            "labels": recon_labels,
        },
        tensor_path,
    )

    if _HAS_TORCHVISION:
        image_path = artifact_dir / f"{prefix}_reconstruction.png"
        save_image(recon_images, image_path, normalize=True, value_range=(0, 1))  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate privacy leakage under RoLoRA-DP.")
    parser.add_argument("--config", required=True, help="YAML config used to build the BiT+LoRA model.")
    parser.add_argument("--payload", required=True, help="Path to serialized gradient payload (.pt or .json).")
    parser.add_argument("--output", default="results_privacy_attack.csv", help="CSV file to append evaluation metrics.")
    parser.add_argument("--artifact-dir", default=None, help="Directory to dump reconstructed tensors/figures.")
    parser.add_argument("--apply_rolora", action="store_true", help="Apply RoLoRA-DP rotation before running the attack.")
    parser.add_argument("--rolora-seed", type=int, default=2025, help="Seed used to generate the orthogonal matrix.")
    parser.add_argument("--rolora-secret-shared", action="store_true", help="If set, attacker knows the rotation and can revert it.")
    parser.add_argument("--attack-steps", type=int, default=2000, help="DLG optimization steps.")
    parser.add_argument("--attack-lr", type=float, default=0.1, help="DLG learning rate.")
    parser.add_argument("--data-reg", type=float, default=1e-4, help="L2 prior weight on reconstructed data.")
    parser.add_argument("--tv-reg", type=float, default=1e-5, help="Total variation regularizer weight.")
    parser.add_argument("--grad-weight", type=float, default=1.0, help="Weight on gradient matching term.")
    parser.add_argument("--label-weight", type=float, default=1e-3, help="Entropy regularization on soft labels.")
    parser.add_argument("--topk", type=int, default=1, help="Top-k singular vectors to keep in SVD stage.")
    parser.add_argument("--device", default="cuda", help="Device to run the attack on (cuda or cpu).")
    parser.add_argument("--num-classes", type=int, default=None, help="Override number of classes for the attack model.")
    parser.add_argument("--config-name", default=None, help="Optional label for the CSV row.")
    parser.add_argument("--dp-epsilon", type=float, default=None, help="Override epsilon for reporting.")
    parser.add_argument("--dp-delta", type=float, default=1e-5, help="Override delta for reporting.")
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint to load before attack.")
    return parser.parse_args()


def extract_b_tensor(payload: Dict) -> Optional[torch.Tensor]:
    if "B_params" not in payload or payload["B_params"] is None:
        return None
    B_section = payload["B_params"]
    if isinstance(B_section, dict):
        tensors = ensure_tensor_dict(B_section)
        first_key = sorted(tensors.keys())[0]
        return tensors[first_key]
    tensor = _tensor_from_payload(B_section)
    return tensor.float() if isinstance(tensor, torch.Tensor) else None


def build_attack_model(config: Dict, batch_images: torch.Tensor, num_classes: int) -> nn.Module:
    try:
        model = make_bit_model_with_lora(config)
        return model
    except Exception as err:
        print(f"⚠️  Falling back to SimpleLoRAModel due to: {err}")
        rank = config.get("model", {}).get("lora", {}).get("r", 8)
        return SimpleLoRAModel(batch_images.shape[1:], num_classes, rank=rank)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    config_path = Path(args.config)
    payload_path = Path(args.payload)
    output_path = Path(args.output)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    payload = load_payload(payload_path)
    if "delta_A_params" not in payload:
        raise ValueError("Payload must contain 'delta_A_params'.")
    delta_A = ensure_tensor_dict(payload["delta_A_params"])
    batch_images = None
    if payload.get("batch_images") is not None:
        batch_images = _tensor_from_payload(payload["batch_images"])
        if isinstance(batch_images, torch.Tensor):
            if batch_images.dtype == torch.uint8:
                batch_images = batch_images.float() / 255.0
            else:
                batch_images = batch_images.float()
    batch_labels = (
        _tensor_from_payload(payload["batch_labels"]).long() if payload.get("batch_labels") is not None else None
    )
    num_classes = args.num_classes or payload.get("num_classes") or config.get("model", {}).get("num_classes")
    if num_classes is None:
        raise ValueError("Number of classes is undefined. Provide --num-classes or include it in the config/payload.")
    if batch_images is None:
        raise ValueError("Payload must include 'batch_images' for evaluation metrics.")

    model = build_attack_model(config, batch_images, num_classes).to(device)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    # Prepare ΔA (with optional RoLoRA rotation)
    attack_delta = delta_A
    rank = next(iter(attack_delta.values())).shape[-1]
    rolora_applied = False
    if args.apply_rolora:
        Q = generate_random_orthogonal_matrix(args.rolora_seed, rank, device=device)
        rotated = apply_rolora_dp(attack_delta, Q)
        rolora_applied = True
        if args.rolora_secret_shared:
            attack_delta = aggregate_with_inverse_rotation([rotated], Q)
        else:
            attack_delta = rotated

    # SVD leakage stage
    b_tensor = extract_b_tensor(payload)
    svd_result, _ = svd_feature_leakage(
        attack_delta,
        B_matrix=b_tensor,
        topk=args.topk,
        device=device,
    )

    attack_config = {
        "attack_lr": args.attack_lr,
        "attack_steps": args.attack_steps,
        "data_reg": args.data_reg,
        "tv_reg": args.tv_reg,
        "grad_weight": args.grad_weight,
        "label_weight": args.label_weight,
        "num_classes": num_classes,
        "optimize_labels": True,
    }

    dlg_result, recon_images, recon_labels = dlg_refinement(
        model,
        attack_delta,
        batch_images=batch_images,
        batch_labels=batch_labels,
        attack_config=attack_config,
        device=device,
    )

    save_reconstruction_artifacts(artifact_dir, recon_images, recon_labels, prefix="rolora_dp")

    row = {
        "config": args.config_name or config_path.stem,
        "dp_epsilon": args.dp_epsilon or payload.get("epsilon"),
        "dp_delta": args.dp_delta,
        "apply_rolora": rolora_applied,
        "rolora_seed": args.rolora_seed if rolora_applied else "",
        "rolora_secret_shared": args.rolora_secret_shared if rolora_applied else "",
        "attack_ssim": dlg_result.ssim,
        "attack_psnr": dlg_result.psnr,
        "attack_mse": dlg_result.reconstruction_mse,
        "attack_label_acc": dlg_result.label_accuracy,
        "gradient_mse": dlg_result.grad_match,
        "singular_energy": svd_result.to_dict()["singular_energy"],
        "top_singular_value": svd_result.to_dict()["top_singular_value"],
        "train_accuracy": payload.get("train_accuracy"),
        "test_accuracy": payload.get("test_accuracy"),
    }
    write_csv_row(output_path, row)

    print("=== Privacy Attack Evaluation ===")
    for key, value in row.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
