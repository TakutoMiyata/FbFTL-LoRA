import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attack_A_reconstruction import reconstruct_A_from_noisy_updates  # noqa: E402


def _make_delta_list(noise_scale=0.0):
    base = {
        "layer": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    deltas = []
    for _ in range(5):
        noisy = {k: v + noise_scale * torch.randn_like(v) for k, v in base.items()}
        deltas.append(noisy)
    return deltas, base


def test_average_attack_recovers_signal_without_noise():
    deltas, base = _make_delta_list(noise_scale=0.0)
    recon = reconstruct_A_from_noisy_updates(deltas, method="average")
    assert torch.allclose(recon["layer"], base["layer"])


def test_svd_attack_reduces_noise():
    deltas, base = _make_delta_list(noise_scale=0.1)
    recon = reconstruct_A_from_noisy_updates(deltas, method="svd")
    error = torch.norm(recon["layer"] - base["layer"])
    assert error < 0.5
