#!/usr/bin/env python3
import os
import yaml
from copy import deepcopy

# === 共通のベース設定 ===
BASE_CONFIG = {
    "seed": 42,
    "use_gpu": True,
    "use_amp": False,
    "data": {
        "dataset_name": "cifar100",
        "data_dir": "./data",
        "num_clients": 10,
        "num_test_clients": 10,
        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "input_size": 224,
        "imagenet_style": True,
        "data_split": "iid",
        "alpha": 1.0,
        "augmentations": {
            "horizontal_flip": {"enabled": True, "prob": 0.5},
            "random_rotation": {"enabled": True, "degrees": 20},
            "random_resized_crop": {"enabled": True, "scale_min": 0.4},
            "color_jitter": {
                "enabled": True, "brightness": 0.3, "contrast": 0.3,
                "saturation": 0.3, "hue": 0.1
            },
            "random_erasing": {"enabled": True, "prob": 0.3},
            "mixup": {"enabled": False, "alpha": 0.2, "prob": 0.5},
            "cutmix": {"enabled": False, "alpha": 1.0, "prob": 0.5},
        },
    },
    "model": {
        "model_name": "bit_s_r50x1",  # 上書き予定
        "num_classes": 100,
        "pretrained": True,
        "freeze_backbone": True,
        "lora": {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "dropout": 0.1
        }
    },
    "training": {
        "epochs": 3,
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.001,
        "scheduler": "cosine",
        "warmup_epochs": 0,
        "label_smoothing": 0.1,
        "gradient_clip": 1.0,
        "optimizer": "sgd",
    },
    "federated": {
        "num_rounds": 100,
        "num_clients": 10,
        "client_fraction": 1.0,
        "aggregation_method": "fedsa",
        "checkpoint_freq": 25,
        "exclude_bn_from_agg": True,
    },
    "privacy": {
        "enable_privacy": False,
        "epsilon": 8.0,
        "delta": 1e-5,
        "max_grad_norm": 0.5,
        "noise_multiplier": 1.0,
        "target": "lora_A",
        "use_opacus_accounting": True,
    },
    "evaluation": {
        "eval_freq": 5,
        "metric": "accuracy",
        "save_best_model": True,
    },
    "experiment": {
        "name": "",
        "output_dir": "",
        "save_history": True,
        "save_model": True,
        "log_interval": 10,
        "use_wandb": False,
        "wandb_project": "",
        "wandb_entity": None,
    },
    "reproducibility": {"deterministic": False},
    "advanced": {
        "personalized": True,
        "personalization_layers": ["lora_B", "head"],
        "use_fbftl": False,
        "feature_extraction_rounds": 10,
        "server_lr": 1.0,
        "server_momentum": 0.9,
    },
    "communication": {
        "compress": False,
        "compression_ratio": 0.1,
        "quantization_bits": None,
    }
}

# === パラメータ空間 ===
methods = ["fedavg", "fedsa_lora", "fedsa_lora_dp"]
models = ["bit_s_r50x1", "bit_s_r101x1"]  # ← r50とr101両方
alphas = [1.0, 0.1]
client_fractions = [1.0, 0.3]
rs = [4, 8]      # クライアント数比較では r=8 を主軸
epsilons = [4, 8]
output_root = "configs/experiment_configs_bit"

# === YAML保存関数 ===
def save_yaml(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"[+] Saved {path}")

# === 出力ディレクトリ命名関数 ===
def make_output_dir(method, model, alpha, f, r=None, eps=None):
    parts = [
        "experiments", "bit", method,
        model,
        f"alpha{alpha}",
        f"f{f}"
    ]
    if r is not None:
        parts.append(f"r{r}")
    if eps is not None:
        parts.append(f"eps{eps}")
    return os.path.join(*parts)

# === 生成ループ ===
for method in methods:
    for model in models:
        for alpha in alphas:
            for f in client_fractions:

                # FedAvg
                if method == "fedavg":
                    cfg = deepcopy(BASE_CONFIG)
                    cfg["model"]["model_name"] = model
                    cfg["model"]["lora"]["enabled"] = False
                    cfg["model"]["freeze_backbone"] = False
                    cfg["federated"]["aggregation_method"] = "fedavg"
                    cfg["data"]["data_split"] = "iid" if alpha == 1.0 else "non_iid"
                    cfg["data"]["alpha"] = alpha
                    cfg["federated"]["client_fraction"] = f
                    cfg["privacy"]["enable_privacy"] = False
                    cfg["experiment"]["name"] = f"{model}_fedavg_alpha{alpha}_f{f}"
                    cfg["experiment"]["output_dir"] = make_output_dir("fedavg", model, alpha, f)
                    cfg["experiment"]["wandb_project"] = f"{model}-fedavg"
                    filename = f"{model}_alpha{alpha}_f{f}_fedavg.yaml"
                    save_yaml(cfg, os.path.join(output_root, "fedavg", filename))

                # FedSA-LoRA
                elif method == "fedsa_lora":
                    for r in rs:
                        cfg = deepcopy(BASE_CONFIG)
                        cfg["model"]["model_name"] = model
                        cfg["model"]["lora"]["enabled"] = True
                        cfg["model"]["lora"]["r"] = r
                        cfg["model"]["freeze_backbone"] = True
                        cfg["federated"]["aggregation_method"] = "fedsa"
                        cfg["data"]["data_split"] = "iid" if alpha == 1.0 else "non_iid"
                        cfg["data"]["alpha"] = alpha
                        cfg["federated"]["client_fraction"] = f
                        cfg["privacy"]["enable_privacy"] = False
                        cfg["experiment"]["name"] = f"{model}_fedsa_lora_alpha{alpha}_f{f}_r{r}"
                        cfg["experiment"]["output_dir"] = make_output_dir("fedsa_lora", model, alpha, f, r=r)
                        cfg["experiment"]["wandb_project"] = f"{model}-fedsa-lora"
                        filename = f"{model}_alpha{alpha}_f{f}_r{r}_fedsa_lora.yaml"
                        save_yaml(cfg, os.path.join(output_root, "fedsa_lora", filename))

                # FedSA-LoRA+DP
                elif method == "fedsa_lora_dp":
                    for r in rs:
                        for eps in epsilons:
                            if alpha == 1.0:
                                # IID+DPはスキップ（必要なら外す）
                                continue
                            cfg = deepcopy(BASE_CONFIG)
                            cfg["model"]["model_name"] = model
                            cfg["model"]["lora"]["enabled"] = True
                            cfg["model"]["lora"]["r"] = r
                            cfg["model"]["freeze_backbone"] = True
                            cfg["federated"]["aggregation_method"] = "fedsa"
                            cfg["data"]["data_split"] = "non_iid"
                            cfg["data"]["alpha"] = alpha
                            cfg["federated"]["client_fraction"] = f
                            cfg["privacy"]["enable_privacy"] = True
                            cfg["privacy"]["epsilon"] = eps
                            cfg["experiment"]["name"] = f"{model}_fedsa_lora_dp_alpha{alpha}_f{f}_r{r}_eps{eps}"
                            cfg["experiment"]["output_dir"] = make_output_dir("fedsa_lora_dp", model, alpha, f, r=r, eps=eps)
                            cfg["experiment"]["wandb_project"] = f"{model}-fedsa-lora-dp"
                            filename = f"{model}_alpha{alpha}_f{f}_r{r}_eps{eps}_fedsa_lora_dp.yaml"
                            save_yaml(cfg, os.path.join(output_root, "fedsa_lora_dp", filename))