#!/usr/bin/env python3
"""
Standalone evaluation script for A-matrix reconstruction attacks.

Expected input format (torch .pt file):
{
  "observed": {client_id: [delta_dict, ...]},
  "true": {client_id: [delta_dict, ...]},
  "config": {...}
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from attack_A_reconstruction import reconstruct_A_from_noisy_updates
from utils.metrics_A_eval import evaluate_A_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A-matrix reconstruction leakage.")
    parser.add_argument("--data", required=True, help="Path to .pt file containing observations and ground truth.")
    parser.add_argument("--method", default="svd", choices=["average", "svd", "dlg"], help="Attack reconstruction method.")
    parser.add_argument("--rounds-used", type=int, default=None, help="Number of rounds to use from the tail.")
    parser.add_argument("--output", default=None, help="Optional path to save metrics JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = torch.load(args.data, map_location="cpu")
    observed_map: Dict[str, list] = payload["observed"]
    true_map: Dict[str, list] = payload["true"]

    attack_cfg = {
        "rounds_used": args.rounds_used,
    }

    all_metrics = {}

    for client_id, observed_list in observed_map.items():
        true_list = true_map.get(client_id)
        if not true_list:
            continue
        recon = reconstruct_A_from_noisy_updates(observed_list, method=args.method, attack_config=attack_cfg)
        metrics = evaluate_A_similarity(recon, true_list[-1])
        all_metrics[client_id] = metrics
        print(f"Client {client_id}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
