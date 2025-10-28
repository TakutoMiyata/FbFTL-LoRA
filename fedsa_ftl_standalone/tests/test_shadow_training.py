import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shadow_training import ShadowDecoder, flatten_delta_dict  # noqa: E402


def test_flatten_delta_dict_orders_keys_deterministically():
    delta = {
        'b': torch.ones(2, 3),
        'a': torch.zeros(1, 3),
    }
    flat = flatten_delta_dict(delta)
    assert flat.shape[0] == 2 * 3 + 1 * 3
    # First part corresponds to key 'a' (zeros), followed by ones from key 'b'
    assert torch.allclose(flat[:3], torch.zeros(3))
    assert torch.allclose(flat[3:], torch.ones(6))


def test_shadow_decoder_output_shape_matches_image_shape():
    decoder = ShadowDecoder(input_dim=5, image_shape=(2, 3, 4, 4), hidden_dim=16)
    vec = torch.randn(7, 5)
    out = decoder(vec)
    assert out.shape == (7, 2, 3, 4, 4)
