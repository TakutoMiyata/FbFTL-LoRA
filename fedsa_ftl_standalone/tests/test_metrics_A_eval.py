import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics_A_eval import evaluate_A_similarity  # noqa: E402


def test_metrics_detect_perfect_alignment():
    true = {"layer": torch.randn(3, 2)}
    recon = {"layer": true["layer"].clone()}
    metrics = evaluate_A_similarity(recon, true)
    assert abs(metrics["nmse_raw"]) < 1e-6
    assert abs(metrics["cos_raw"] - 1.0) < 1e-6


def test_metrics_handle_orthogonal_rotation():
    torch.manual_seed(0)
    true = {"layer": torch.randn(4, 2)}
    Q, _ = torch.linalg.qr(torch.randn(2, 2))
    recon = {"layer": true["layer"] @ Q}
    metrics = evaluate_A_similarity(recon, true)
    assert metrics["cos_raw"] < 1.0
    assert metrics["cos_aligned"] > 0.99
