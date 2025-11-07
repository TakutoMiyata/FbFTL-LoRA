#!/usr/bin/env python3
"""
Quick start script for Dual-B FedSA-LoRA with RL-updated server matrices and Shapley analysis.

This implementation follows the design sketched in next_jornal.md:
    - Clients train local LoRA A/B matrices on their own data.
    - Server samples stochastic perturbations of a shared B_server matrix and
      receives only scalar rewards (forward-only evaluation) from clients.
    - Policy-gradient style updates (REINFORCE/SPSA) adjust B_server.
    - Optional Monte-Carlo Shapley estimation quantifies client contribution.

The script reuses the ViT-based FedSA-FTL model defined in src/fedsa_ftl_model_vit.py.
"""

import argparse
import copy
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import prepare_federated_data
from fedsa_ftl_client import FedSAFTLClient
from fedsa_ftl_model_vit import create_model_vit


os.environ.setdefault("TQDM_DISABLE", "0")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloader(dataset, indices, batch_size, num_workers=0, shuffle=True):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def detach_to_cpu(tensor_dict):
    return {k: v.detach().cpu() for k, v in tensor_dict.items()}


def zero_lora_B_parameters(model):
    for module in model.modules():
        if hasattr(module, "lora_B"):
            module.lora_B.data.zero_()


class DualBFedSAFTLClient(FedSAFTLClient):
    """Client that can temporarily inject server-controlled B matrices for evaluation."""

    def __init__(self, client_id, model, device, reward_loader, reward_max_batches=None):
        super().__init__(client_id, model, device)
        self.reward_loader = reward_loader
        self.reward_max_batches = reward_max_batches
        self._criterion = nn.CrossEntropyLoss()

    def clear_server_B(self):
        if hasattr(self.model, "reset_server_B_params"):
            self.model.reset_server_B_params()

    def set_server_B(self, server_B_params):
        if server_B_params and hasattr(self.model, "set_server_B_params"):
            self.model.set_server_B_params(server_B_params)
        else:
            self.clear_server_B()

    def evaluate_server_reward(self):
        if self.reward_loader is None:
            return {"reward": 0.0, "loss": 0.0, "samples": 0}

        self.model.eval()
        total = 0
        correct = 0
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

        reward = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return {"reward": reward, "loss": avg_loss, "samples": total}


class DualBFedSAServer:
    """Server managing global A matrices, RL-style B_server, and Shapley estimation."""

    def __init__(self, model_template, device, dual_b_cfg, shapley_cfg, eval_loader=None):
        self.device = device
        self.dual_b_cfg = dual_b_cfg or {}
        self.noise_std = float(self.dual_b_cfg.get("noise_std", 0.01))
        self.server_lr = float(self.dual_b_cfg.get("server_lr", 0.1))
        self.baseline_beta = float(self.dual_b_cfg.get("baseline_beta", 0.1))
        self.max_reward_norm = float(self.dual_b_cfg.get("max_reward_norm", 5.0))
        self.global_round = 0

        self.global_A_params = detach_to_cpu(model_template.get_lora_params(matrix_type="A"))
        self.zero_A_template = {k: torch.zeros_like(v) for k, v in self.global_A_params.items()}
        self.server_B_params = self._init_server_B_buffers(model_template)

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
            self.eval_model = copy.deepcopy(model_template).to(self.device)
            zero_lora_B_parameters(self.eval_model)
            self.eval_model.eval()

        if bool(self.shapley_cfg.get("enabled", False)) and self.eval_loader is None:
            print("⚠️  Shapley estimation requested but no eval loader provided. Disabled.")

    @staticmethod
    def _init_server_B_buffers(model_template):
        buffers = {}
        for name, module in model_template.named_modules():
            if hasattr(module, "lora_B"):
                key = f"{name}.lora_B_server"
                buffers[key] = torch.zeros_like(module.lora_B.detach().cpu())
        return buffers

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

    def aggregate_A_matrices(self, client_updates):
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

    def apply_client_reward(self, client_id, reward, weight):
        if client_id not in self.noise_cache or not self.server_B_params:
            return

        noise = self.noise_cache.pop(client_id)
        baseline = self.reward_baseline if self.reward_baseline is not None else reward
        advantage = reward - baseline

        # Update running reward baseline
        beta = self.baseline_beta
        self.reward_baseline = (1 - beta) * baseline + beta * reward

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
        if not self.server_B_params or self.pending_b_grad is None:
            return

        normalizer = max(self.total_reward_weight, 1.0)
        for name in self.server_B_params.keys():
            grad = self.pending_b_grad[name] / normalizer
            self.server_B_params[name] += self.server_lr * grad

        self.pending_b_grad = None
        self.total_reward_weight = 0.0

    def evaluate_subset_accuracy(self, aggregated_A):
        if self.eval_model is None or self.eval_loader is None:
            return 0.0

        self.eval_model.set_lora_params(aggregated_A or self.zero_A_template, matrix_type="A")
        self.eval_model.reset_server_B_params()
        self.eval_model.set_server_B_params(self.server_B_params)

        total = 0
        correct = 0

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


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def maybe_create_default_config(config_path: Path):
    if config_path.exists():
        return

    default_config = {
        "seed": 42,
        "use_gpu": True,
        "data": {
            "dataset_name": "cifar100",
            "data_dir": "./data",
            "num_clients": 4,
            "batch_size": 32,
            "num_workers": 2,
            "data_split": "non_iid",
            "alpha": 0.5,
            "verbose": False,
            "imagenet_style": True,
            "input_size": 224,
        },
        "model": {
            "model_name": "vit_tiny",
            "num_classes": 100,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "freeze_backbone": True,
        },
        "training": {
            "local_epochs": 1,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
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
            "reward_batches": 2,
        },
        "shapley": {
            "enabled": True,
            "samples": 3,
            "eval_batches": 2,
            "min_clients": 3,
        },
        "evaluation": {
            "server_eval_samples": 512,
            "batch_size": 64,
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(default_config, f)
    print(f"✅ Default config created at {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Dual-B FedSA-LoRA quickstart (ViT)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dual_b_vit_cifar100.yaml",
        help="Path to YAML config",
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

    num_clients = int(data_cfg.get("num_clients", 4))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))
    reward_max_batches = dual_b_cfg.get("reward_batches")

    print(f"Preparing data ({data_cfg.get('dataset_name', 'cifar100')}) ...")
    trainset, testset, client_train_indices, client_test_indices = prepare_federated_data(
        {
            **data_cfg,
            "num_clients": num_clients,
        }
    )

    server_eval_loader = None
    eval_samples = int(eval_cfg.get("server_eval_samples", 0))
    if eval_samples > 0:
        eval_indices = list(range(len(testset)))
        random.shuffle(eval_indices)
        eval_indices = eval_indices[: min(eval_samples, len(eval_indices))]
        server_eval_loader = build_dataloader(
            testset,
            eval_indices,
            batch_size=eval_cfg.get("batch_size", 64),
            num_workers=num_workers,
            shuffle=False,
        )

    print("Initializing models ...")
    base_model = create_model_vit(
        {
            "num_classes": model_cfg.get("num_classes", 100),
            "model_name": model_cfg.get("model_name", "vit_tiny"),
            "lora_r": model_cfg.get("lora_r", 8),
            "lora_alpha": model_cfg.get("lora_alpha", 16),
            "lora_dropout": model_cfg.get("lora_dropout", 0.05),
            "freeze_backbone": model_cfg.get("freeze_backbone", True),
        }
    )
    base_model.to(device)

    clients = []
    for client_id in range(num_clients):
        client_model = copy.deepcopy(base_model)
        client_model.to(device)

        reward_loader = build_dataloader(
            testset,
            client_test_indices[client_id],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        client = DualBFedSAFTLClient(
            client_id,
            client_model,
            device,
            reward_loader,
            reward_max_batches=reward_max_batches,
        )
        clients.append(client)

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

    print(f"Starting Dual-B federated training for {total_rounds} rounds ...")
    for round_idx in range(total_rounds):
        round_start = time.time()
        num_selected = max(1, math.ceil(client_fraction * num_clients))
        selected = sorted(random.sample(range(num_clients), num_selected))

        print(f"\n[Round {round_idx + 1}/{total_rounds}] Selected clients: {selected}")

        client_updates = []
        train_metrics = []

        for client_id in selected:
            client = clients[client_id]

            client.update_model({"A_params": server.global_A_params})
            client.clear_server_B()

            train_loader = build_dataloader(
                trainset,
                client_train_indices[client_id],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )

            update = client.train(train_loader, training_cfg)
            update["lora_A_params"] = detach_to_cpu(update["lora_A_params"])

            perturbed_B = server.sample_perturbed_B(client_id)
            client.set_server_B(perturbed_B)
            reward_info = client.evaluate_server_reward()
            client.clear_server_B()

            update["reward"] = reward_info["reward"]
            update["reward_loss"] = reward_info["loss"]
            client_updates.append(update)
            train_metrics.append((update["loss"], update["accuracy"]))

            server.apply_client_reward(client_id, reward_info["reward"], update["num_samples"])

        server.finalize_b_update()
        aggregated_A = server.aggregate_A_matrices(client_updates)
        shapley_scores = server.estimate_shapley(client_updates)

        avg_loss = np.mean([m[0] for m in train_metrics]) if train_metrics else 0.0
        avg_acc = np.mean([m[1] for m in train_metrics]) if train_metrics else 0.0
        avg_reward = np.mean([u["reward"] for u in client_updates]) if client_updates else 0.0
        comm_cost_mb = sum(
            tensor.numel() * 4 for tensor in aggregated_A.values()
        ) / (1024 * 1024)

        round_duration = time.time() - round_start
        history.append(
            {
                "round": round_idx + 1,
                "clients": len(selected),
                "train_loss": avg_loss,
                "train_accuracy": avg_acc,
                "reward": avg_reward,
                "comm_mb": comm_cost_mb,
                "duration_s": round_duration,
                "shapley": shapley_scores,
            }
        )

        print(
            f"Round {round_idx + 1}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%, "
            f"reward={avg_reward:.3f}, comm={comm_cost_mb:.2f}MB, time={round_duration:.1f}s"
        )
        if shapley_scores:
            top = sorted(shapley_scores.items(), key=lambda kv: kv[1], reverse=True)
            summary = ", ".join([f"C{cid}:{score:.3f}" for cid, score in top[:3]])
            print(f"  Shapley (top-3): {summary}")

    print("\nTraining complete. Summary of last round:")
    if history:
        last = history[-1]
        print(last)


if __name__ == "__main__":
    main()
