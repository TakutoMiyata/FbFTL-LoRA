#!/usr/bin/env python3
"""
Dual-B FedSA-LoRA quickstart for MNIST/SVHN with ViT-style backbones.

Clients train local LoRA A/B matrices on a ViT (or optional compact CNN) while the server
maintains a shared B_server updated via stochastic policy gradients from forward-only
rewards. Optional Monte-Carlo Shapley estimation evaluates client contributions.
"""

import argparse
import copy
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fedsa_ftl_client import FedSAFTLClient
from dual_b_small_cnn import SmallConvNetLoRA
from fedsa_ftl_model_vit import create_model_vit


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(dataset_name: str, input_size: int, num_channels: int, train: bool):
    ops: List[transforms.Compose] = []
    if dataset_name == "mnist":
        ops.append(transforms.Grayscale(num_output_channels=num_channels))
    else:
        if num_channels == 1:
            ops.append(transforms.Grayscale(num_output_channels=1))
        else:
            ops.append(transforms.Lambda(lambda img: img.convert("RGB")))

    if train:
        ops.append(transforms.Resize((input_size + 8, input_size + 8)))
        ops.append(
            transforms.RandomCrop(
                input_size,
                padding=0,
            )
        )
        ops.append(transforms.RandomHorizontalFlip())
    else:
        ops.append(transforms.Resize((input_size, input_size)))

    ops.append(transforms.ToTensor())
    mean = [0.5] * num_channels
    std = [0.5] * num_channels
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def load_dataset(data_cfg: Dict, num_channels: int):
    dataset_name = data_cfg.get("dataset_name", "mnist").lower()
    input_size = int(data_cfg.get("input_size", 64))
    data_dir = data_cfg.get("data_dir", "./data")

    train_transform = build_transforms(dataset_name, input_size, num_channels, train=True)
    test_transform = build_transforms(dataset_name, input_size, num_channels, train=False)

    if dataset_name == "mnist":
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    elif dataset_name == "svhn":
        trainset = torchvision.datasets.SVHN(root=data_dir, split="train", download=True, transform=train_transform)
        testset = torchvision.datasets.SVHN(root=data_dir, split="test", download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return trainset, testset


def _extract_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise ValueError("Dataset must expose targets or labels for partitioning")
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    return np.array(targets)


def dirichlet_partition(targets: np.ndarray, num_clients: int, alpha: float) -> List[np.ndarray]:
    num_classes = targets.max() + 1
    class_indices = [np.where(targets == cls)[0] for cls in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = class_indices[cls]
        if len(cls_idx) == 0:
            continue
        np.random.shuffle(cls_idx)
        proportions = np.random.dirichlet(np.full(num_clients, alpha))
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)
        start = 0
        for client_id, end in enumerate(proportions):
            client_indices[client_id].extend(cls_idx[start:end])
            start = end

    return [np.array(sorted(idx), dtype=np.int64) for idx in client_indices]


def iid_partition(num_samples: int, num_clients: int) -> List[np.ndarray]:
    indices = np.random.permutation(num_samples)
    splits = np.array_split(indices, num_clients)
    return [split.astype(np.int64) for split in splits]


def prepare_federated_partitions(trainset, testset, data_cfg: Dict):
    num_clients = int(data_cfg.get("num_clients", 4))
    alpha = float(data_cfg.get("alpha", 0.5))
    split_mode = data_cfg.get("data_split", "non_iid").lower()

    train_targets = _extract_targets(trainset)
    test_targets = _extract_targets(testset)

    if split_mode == "iid":
        train_splits = iid_partition(len(trainset), num_clients)
        test_splits = iid_partition(len(testset), num_clients)
    else:
        train_splits = dirichlet_partition(train_targets, num_clients, alpha)
        test_splits = dirichlet_partition(test_targets, num_clients, alpha)

    return train_splits, test_splits


def build_loader(dataset, indices: Sequence[int], batch_size: int, num_workers: int, shuffle: bool):
    subset = Subset(dataset, indices.tolist() if isinstance(indices, np.ndarray) else list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_model(model_cfg: Dict):
    backbone = model_cfg.get("backbone", "vit").lower()
    if backbone == "vit":
        vit_cfg = {
            "num_classes": model_cfg.get("num_classes", 10),
            "model_name": model_cfg.get("model_name", "vit_tiny"),
            "lora_r": model_cfg.get("lora_r", 8),
            "lora_alpha": model_cfg.get("lora_alpha", 16),
            "lora_dropout": model_cfg.get("lora_dropout", 0.1),
            "freeze_backbone": model_cfg.get("freeze_backbone", True),
            "input_size": model_cfg.get("input_size", 160),
        }
        return create_model_vit(vit_cfg)
    elif backbone == "small_cnn":
        return SmallConvNetLoRA(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 10),
            input_size=model_cfg.get("input_size", 64),
            hidden_channels=model_cfg.get("hidden_channels", 64),
            lora_r=model_cfg.get("lora_r", 4),
            lora_alpha=model_cfg.get("lora_alpha", 16),
            lora_dropout=model_cfg.get("lora_dropout", 0.05),
        )
    else:
        raise ValueError(f"Unsupported backbone '{backbone}' for Dual-B quickstart")


class DualBFedSAFTLClient(FedSAFTLClient):
    """Client that can inject server B matrices for reward-only evaluation."""

    def __init__(self, client_id, model, device, reward_loader=None, reward_max_batches=None):
        super().__init__(client_id, model, device)
        self.reward_loader = reward_loader
        self.reward_max_batches = reward_max_batches
        self._criterion = nn.CrossEntropyLoss()
        self._cached_local_B = None

    def set_server_B(self, server_B_params):
        if self._cached_local_B is None and hasattr(self.model, "get_lora_params"):
            self._cached_local_B = self.model.get_lora_params(matrix_type="B")
            zero_params = {k: torch.zeros_like(v) for k, v in self._cached_local_B.items()}
            self.model.set_lora_params(zero_params, matrix_type="B")
        if server_B_params and hasattr(self.model, "set_server_B_params"):
            self.model.set_server_B_params(server_B_params)
        elif hasattr(self.model, "reset_server_B_params"):
            self.model.reset_server_B_params()

    def clear_server_B(self):
        if hasattr(self.model, "reset_server_B_params"):
            self.model.reset_server_B_params()
        if self._cached_local_B is not None:
            self.model.set_lora_params(self._cached_local_B, matrix_type="B")
            self._cached_local_B = None

    def evaluate_server_reward(self):
        if self.reward_loader is None:
            return {"reward": 0.0, "loss": 0.0, "samples": 0}

        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.reward_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self._criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                if self.reward_max_batches and (batch_idx + 1) >= self.reward_max_batches:
                    break

        reward = (correct / total) if total > 0 else 0.0
        avg_loss = (total_loss / total) if total > 0 else 0.0
        return {"reward": reward, "loss": avg_loss, "samples": total}


def detach_to_cpu(tensor_dict):
    return {k: v.detach().cpu() for k, v in tensor_dict.items()}


class DualBFedSAServer:
    """Server that aggregates A matrices and updates shared B_server via SPSA/REINFORCE."""

    def __init__(self, model_template, device, dual_b_cfg, shapley_cfg, eval_loader=None):
        self.device = device
        self.dual_b_cfg = dual_b_cfg or {}
        self.noise_std = float(self.dual_b_cfg.get("noise_std", 0.02))
        self.server_lr = float(self.dual_b_cfg.get("server_lr", 0.05))
        self.baseline_beta = float(self.dual_b_cfg.get("baseline_beta", 0.1))
        self.max_reward_norm = float(self.dual_b_cfg.get("max_reward_norm", 5.0))

        self.global_A_params = detach_to_cpu(model_template.get_lora_params(matrix_type="A"))
        self.zero_A_template = {k: torch.zeros_like(v) for k, v in self.global_A_params.items()}
        self.server_B_params = model_template.get_server_B_params()

        self.noise_cache = {}
        self.pending_b_grad = None
        self.total_reward_weight = 0.0
        self.reward_baseline = None

        self.eval_loader = eval_loader
        self.shapley_cfg = shapley_cfg or {}
        self.shapley_enabled = bool(self.shapley_cfg.get("enabled", False)) and self.eval_loader is not None
        self.shapley_samples = int(self.shapley_cfg.get("samples", 4))
        self.shapley_eval_batches = int(self.shapley_cfg.get("eval_batches", 2))
        self.shapley_min_clients = int(self.shapley_cfg.get("min_clients", 2))

        self.eval_model = None
        if self.eval_loader is not None:
            self.eval_model = copy.deepcopy(model_template).to(device)
            self.eval_model.reset_server_B_params()

    def sample_perturbed_B(self, client_id):
        if not self.server_B_params:
            return {}

        noise = {}
        perturbed = {}
        for name, base in self.server_B_params.items():
            rand = torch.randn_like(base) * self.noise_std
            noise[name] = rand
            perturbed[name] = base + rand

        self.noise_cache[client_id] = noise
        return perturbed

    def apply_client_reward(self, client_id, reward, weight):
        if client_id not in self.noise_cache or not self.server_B_params:
            return

        noise = self.noise_cache.pop(client_id)
        baseline = reward if self.reward_baseline is None else self.reward_baseline
        advantage = reward - baseline
        self.reward_baseline = (1 - self.baseline_beta) * baseline + self.baseline_beta * reward

        if self.pending_b_grad is None:
            self.pending_b_grad = {name: torch.zeros_like(param) for name, param in self.server_B_params.items()}
            self.total_reward_weight = 0.0

        clip = self.max_reward_norm
        scaled_advantage = max(-clip, min(clip, advantage))
        weight = max(weight, 1.0)
        self.total_reward_weight += weight

        denom = self.noise_std if self.noise_std > 0 else 1.0
        for name in self.server_B_params.keys():
            self.pending_b_grad[name] += (noise[name] * (scaled_advantage * weight)) / denom

    def finalize_b_update(self):
        if self.pending_b_grad is None:
            return

        normalizer = max(self.total_reward_weight, 1.0)
        for name in self.server_B_params.keys():
            grad = self.pending_b_grad[name] / normalizer
            self.server_B_params[name] += self.server_lr * grad

        self.pending_b_grad = None
        self.total_reward_weight = 0.0

    def aggregate_A_matrices(self, client_updates: List[Dict]):
        if not client_updates:
            return self.global_A_params

        total_samples = sum(update["num_samples"] for update in client_updates)
        aggregated = {name: torch.zeros_like(param) for name, param in self.global_A_params.items()}

        for update in client_updates:
            weight = update["num_samples"] / total_samples if total_samples > 0 else 1.0 / len(client_updates)
            for name, tensor in update["lora_A_params"].items():
                aggregated[name] += tensor * weight

        self.global_A_params = aggregated
        return aggregated

    def evaluate_subset_accuracy(self, aggregated_A):
        if self.eval_model is None or self.eval_loader is None:
            return 0.0

        self.eval_model.set_lora_params(aggregated_A or self.zero_A_template, matrix_type="A")
        self.eval_model.reset_server_B_params()
        self.eval_model.set_server_B_params(self.server_B_params)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.eval_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.eval_model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                if self.shapley_eval_batches and (batch_idx + 1) >= self.shapley_eval_batches:
                    break

        return (correct / total) if total > 0 else 0.0

    def _aggregate_subset_A(self, client_updates, subset_indices):
        if not subset_indices:
            return self.zero_A_template
        total_samples = sum(client_updates[idx]["num_samples"] for idx in subset_indices)
        aggregated = {name: torch.zeros_like(param) for name, param in self.global_A_params.items()}
        for idx in subset_indices:
            update = client_updates[idx]
            weight = update["num_samples"] / total_samples if total_samples > 0 else 1.0 / len(subset_indices)
            for name, tensor in update["lora_A_params"].items():
                aggregated[name] += tensor * weight
        return aggregated

    def estimate_shapley(self, client_updates):
        if not self.shapley_enabled or len(client_updates) < self.shapley_min_clients:
            return {}

        client_indices = list(range(len(client_updates)))
        shapley_scores = {idx: 0.0 for idx in client_indices}

        for _ in range(self.shapley_samples):
            permutation = client_indices.copy()
            random.shuffle(permutation)
            subset = []
            previous_value = 0.0

            for idx in permutation:
                subset.append(idx)
                aggregated = self._aggregate_subset_A(client_updates, subset)
                current_value = self.evaluate_subset_accuracy(aggregated)
                shapley_scores[idx] += current_value - previous_value
                previous_value = current_value

        for idx in shapley_scores.keys():
            shapley_scores[idx] /= max(1, self.shapley_samples)

        return shapley_scores


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def maybe_create_default_config(config_path: Path):
    if config_path.exists():
        return

    default_config = {
        "seed": 42,
        "use_gpu": True,
        "data": {
            "dataset_name": "mnist",
            "data_dir": "./data",
            "num_clients": 4,
            "batch_size": 32,
            "num_workers": 2,
            "data_split": "non_iid",
            "alpha": 0.5,
            "input_size": 160,
        },
        "model": {
            "backbone": "vit",
            "model_name": "vit_tiny",
            "num_classes": 10,
            "in_channels": 3,
            "input_size": 160,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "freeze_backbone": True,
        },
        "training": {
            "local_epochs": 1,
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
            "optimizer": "adam",
        },
        "federated": {
            "num_rounds": 5,
            "client_fraction": 1.0,
        },
        "dual_b": {
            "server_lr": 0.05,
            "noise_std": 0.01,
            "baseline_beta": 0.1,
            "max_reward_norm": 2.0,
            "reward_batches": 1,
        },
        "shapley": {
            "enabled": False,
            "samples": 3,
            "eval_batches": 2,
            "min_clients": 3,
        },
        "evaluation": {
            "server_eval_samples": 256,
            "batch_size": 64,
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(default_config, f)
    print(f"✅ Default config created at {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Dual-B FedSA-LoRA quickstart (MNIST/SVHN, ViT backbone)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dual_b_mnist_svhn.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    maybe_create_default_config(config_path)
    config = load_config(config_path)

    set_seed(config.get("seed", 42))
    use_gpu = config.get("use_gpu", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    federated_cfg = config.get("federated", {})
    dual_b_cfg = config.get("dual_b", {})
    shapley_cfg = config.get("shapley", {})
    eval_cfg = config.get("evaluation", {})

    dataset_name = data_cfg.get("dataset_name", "mnist").lower()
    backbone_type = model_cfg.get("backbone", "vit").lower()
    default_channels = 3 if backbone_type == "vit" else (1 if dataset_name == "mnist" else 3)
    num_channels = int(model_cfg.get("in_channels", default_channels))
    model_cfg["in_channels"] = num_channels
    default_input = model_cfg.get(
        "input_size",
        data_cfg.get("input_size", 160 if backbone_type == "vit" else 64)
    )
    model_cfg["input_size"] = default_input
    data_cfg["input_size"] = int(data_cfg.get("input_size", default_input))
    model_cfg.setdefault("num_classes", 10 if dataset_name == "mnist" else 10)

    print(
        f"Loading {dataset_name.upper()} dataset with {num_channels} channel(s) "
        f"at {data_cfg['input_size']}x{data_cfg['input_size']} for backbone '{backbone_type}' ..."
    )
    trainset, testset = load_dataset(data_cfg, num_channels)
    train_splits, test_splits = prepare_federated_partitions(trainset, testset, data_cfg)

    base_model = build_model(model_cfg).to(device)

    num_clients = int(data_cfg.get("num_clients", 4))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))
    reward_batches = dual_b_cfg.get("reward_batches", 1)

    clients = []
    client_eval_indices = {}

    for client_id in range(num_clients):
        client_model = copy.deepcopy(base_model).to(device)
        reward_loader = build_loader(
            testset,
            test_splits[client_id],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        clients.append(
            DualBFedSAFTLClient(
                client_id,
                client_model,
                device,
                reward_loader=reward_loader,
                reward_max_batches=reward_batches,
            )
        )
        client_eval_indices[client_id] = test_splits[client_id]

    server_eval_loader = None
    eval_samples = int(eval_cfg.get("server_eval_samples", 0))
    if eval_samples > 0:
        eval_indices = np.random.permutation(len(testset))[:eval_samples]
        server_eval_loader = build_loader(
            testset,
            eval_indices,
            batch_size=eval_cfg.get("batch_size", 64),
            num_workers=num_workers,
            shuffle=False,
        )

    server = DualBFedSAServer(
        model_template=base_model,
        device=device,
        dual_b_cfg=dual_b_cfg,
        shapley_cfg=shapley_cfg,
        eval_loader=server_eval_loader,
    )

    total_rounds = int(federated_cfg.get("num_rounds", 5))
    client_fraction = float(federated_cfg.get("client_fraction", 1.0))
    history = []

    print(f"Starting Dual-B federated training for {total_rounds} rounds (client fraction {client_fraction}) ...")
    for round_idx in range(total_rounds):
        round_start = time.time()
        num_selected = max(1, math.ceil(client_fraction * num_clients))
        selected = sorted(random.sample(range(num_clients), num_selected))
        print(f"\n[Round {round_idx + 1}/{total_rounds}] Selected clients: {selected}")

        client_updates = []
        train_metrics = []
        client_eval_stats = {}

        for client_id in selected:
            client = clients[client_id]
            train_loader = build_loader(
                trainset,
                train_splits[client_id],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )

            if server.global_A_params:
                client.update_model(server.global_A_params)

            result = client.train(train_loader, training_cfg)
            result["lora_A_params"] = detach_to_cpu(result["lora_A_params"])
            client_updates.append(result)
            train_metrics.append((result["loss"], result["accuracy"]))

            eval_loader = build_loader(
                testset,
                client_eval_indices[client_id],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
            eval_metrics = client.evaluate(eval_loader)
            client_eval_stats[client_id] = {
                "loss": float(eval_metrics["loss"]),
                "accuracy": float(eval_metrics["accuracy"]),
                "num_samples": int(eval_metrics["num_samples"]),
            }

            perturbed_B = server.sample_perturbed_B(client_id)
            client.set_server_B(perturbed_B)
            reward_info = client.evaluate_server_reward()
            client.clear_server_B()
            result["reward"] = reward_info["reward"]
            server.apply_client_reward(client_id, reward_info["reward"], result["num_samples"])

        server.finalize_b_update()
        aggregated_A = server.aggregate_A_matrices(client_updates)
        shapley_scores = server.estimate_shapley(client_updates)

        avg_loss = np.mean([m[0] for m in train_metrics]) if train_metrics else 0.0
        avg_acc = np.mean([m[1] for m in train_metrics]) if train_metrics else 0.0
        avg_reward = np.mean([u["reward"] for u in client_updates]) if client_updates else 0.0
        avg_eval_acc = np.mean([stats["accuracy"] for stats in client_eval_stats.values()]) if client_eval_stats else 0.0
        comm_cost_mb = sum(param.numel() * 4 for param in aggregated_A.values()) / (1024 * 1024)
        round_duration = time.time() - round_start

        history.append(
            {
                "round": round_idx + 1,
                "clients": len(selected),
                "train_loss": avg_loss,
                "train_accuracy": avg_acc,
                "reward": avg_reward,
                "eval_accuracy": avg_eval_acc,
                "comm_mb": comm_cost_mb,
                "duration_s": round_duration,
                "shapley": shapley_scores,
                "client_eval": client_eval_stats,
            }
        )

        print(
            f"Round {round_idx + 1}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%, "
            f"reward={avg_reward:.3f}, comm={comm_cost_mb:.2f}MB, "
            f"eval_acc={avg_eval_acc:.2f}%, time={round_duration:.1f}s"
        )
        if client_eval_stats:
            eval_summary = ", ".join(
                f"C{cid}:{stats['accuracy']:.1f}%"
                for cid, stats in sorted(client_eval_stats.items())
            )
            print(f"  Client eval acc: {eval_summary}")
        if shapley_scores:
            top = sorted(shapley_scores.items(), key=lambda kv: kv[1], reverse=True)
            summary = ", ".join([f"C{selected[idx]}:{score:.3f}" for idx, score in top[:3]])
            print(f"  Shapley (top-3): {summary}")

    print("\nTraining complete. Last round summary:")
    if history:
        print(history[-1])


if __name__ == "__main__":
    main()
