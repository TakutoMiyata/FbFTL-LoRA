import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rolora_dp import (  # noqa: E402
    aggregate_with_inverse_rotation,
    apply_rolora_dp,
    generate_random_orthogonal_matrix,
)


def test_generate_random_orthogonal_matrix_is_reproducible():
    q1 = generate_random_orthogonal_matrix(99, 4)
    q2 = generate_random_orthogonal_matrix(99, 4)
    assert torch.allclose(q1, q2)
    identity = torch.eye(4)
    assert torch.allclose(q1.T @ q1, identity, atol=1e-6)


def test_rotation_round_trip_preserves_updates():
    torch.manual_seed(0)
    delta = {
        'layer0': torch.randn(3, 2),
        'layer1': torch.randn(5, 2),
    }
    Q = generate_random_orthogonal_matrix(7, 2)
    rotated = apply_rolora_dp(delta, Q)
    recovered = aggregate_with_inverse_rotation([rotated], Q)
    for key in delta:
        assert torch.allclose(delta[key], recovered[key], atol=1e-6)
