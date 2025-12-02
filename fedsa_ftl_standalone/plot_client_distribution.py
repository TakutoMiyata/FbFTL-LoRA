#!/usr/bin/env python3
"""
Visualize per-client label distributions saved by quickstart_bit_tff.py (Pillow-based heatmaps).

Usage:
  python plot_client_distribution.py --json experiments/.../client_data_distribution_1127BiT.json
  # optionally set output path
  python plot_client_distribution.py --json path/to/file.json --out outputs/heatmap.png
"""

import argparse
import json
from pathlib import Path

from PIL import Image


def build_matrix(client_entries, num_classes):
    """Return matrix shape (num_clients, num_classes) with label counts."""
    num_clients = len(client_entries)
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_clients)]
    for row, entry in enumerate(client_entries):
        for cls, count in entry.get("class_counts", {}).items():
            matrix[row][int(cls)] = int(count)
    return matrix


def plot_heatmap(matrix, title, out_path):
    """
    Render a simple heatmap using Pillow to avoid GUI/OMP issues in sandboxed envs.
    Each pixel block corresponds to (client, class) with intensity proportional to counts.
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    max_count = max((max(row) for row in matrix), default=1)

    # Scale up so the image is readable even for small grids
    scale = max(4, min(12, 1200 // max(rows, cols, 1)))
    img = Image.new("RGB", (cols * scale, rows * scale), color=(255, 255, 255))

    for r in range(rows):
        for c in range(cols):
            count = matrix[r][c]
            intensity = int(255 * (count / max_count)) if max_count else 0
            # Simple blue->yellow gradient
            color = (intensity, min(255, intensity + 80), 255 - intensity // 2)
            for y in range(r * scale, (r + 1) * scale):
                for x in range(c * scale, (c + 1) * scale):
                    img.putpixel((x, y), color)

    # Add a small border with the title baked into the filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot client data distribution heatmaps")
    parser.add_argument("--json", required=True, help="Path to client_data_distribution_*.json")
    parser.add_argument("--out", help="Optional output image path")
    parser.add_argument("--prefix", default="client_distribution", help="Prefix for generated files when --out is omitted")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        payload = json.load(f)

    num_classes = int(payload.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("num_classes missing or invalid in JSON")

    base_dir = json_path.parent
    stem = json_path.stem.replace("client_data_distribution_", "")
    prefix = args.prefix or "client_distribution"

    train_matrix = build_matrix(payload.get("train_clients", []), num_classes)
    test_matrix = build_matrix(payload.get("test_clients", []), num_classes)

    if args.out:
        out_base = Path(args.out)
        train_out = out_base
        test_out = out_base.with_name(out_base.stem + "_test" + out_base.suffix)
    else:
        train_out = base_dir / f"{prefix}_{stem}_train.png"
        test_out = base_dir / f"{prefix}_{stem}_test.png"

    train_saved = plot_heatmap(train_matrix, f"Train Distribution ({payload.get('split_method', 'N/A')})", train_out)
    test_saved = plot_heatmap(test_matrix, "Test Distribution", test_out)

    print(f"Saved train heatmap to: {train_saved}")
    print(f"Saved test heatmap to: {test_saved}")


if __name__ == "__main__":
    main()
