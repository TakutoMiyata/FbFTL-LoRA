import pathlib
import sys
import types

import torch


def _ensure_opacus_stub():
    if "opacus" in sys.modules:
        return

    opacus = types.ModuleType("opacus")

    class _DummyPrivacyEngine:
        def __init__(self, *args, **kwargs):
            pass

        def make_private(self, module, optimizer, data_loader, noise_multiplier, max_grad_norm, poisson_sampling=False):
            return module, optimizer, data_loader

        def get_epsilon(self, delta):
            return 0.0

    opacus.PrivacyEngine = _DummyPrivacyEngine

    grad_sample = types.ModuleType("opacus.grad_sample")
    utils = types.ModuleType("opacus.grad_sample.utils")

    def _clear_grad_sample(model):
        return None

    utils.clear_grad_sample = _clear_grad_sample
    grad_sample.utils = utils
    grad_sample.clear_grad_sample = _clear_grad_sample

    sys.modules["opacus"] = opacus
    sys.modules["opacus.grad_sample"] = grad_sample
    sys.modules["opacus.grad_sample.utils"] = utils


_ensure_opacus_stub()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dual_b_small_cnn import SmallConvNetLoRA  # noqa: E402
from quickstart_dual_b_lora import DualBFedSAServer  # noqa: E402


def test_small_conv_lora_forward_shape():
    model = SmallConvNetLoRA(in_channels=1, num_classes=5, input_size=64, hidden_channels=16)
    x = torch.randn(2, 1, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 5)


def test_server_b_roundtrip_and_reset():
    model = SmallConvNetLoRA(in_channels=3, num_classes=4, input_size=64, hidden_channels=8, lora_r=2)
    params = model.get_server_B_params()
    assert params, "Server B params should not be empty when LoRA rank > 0"

    mutated = {k: v + 1.0 for k, v in params.items()}
    model.set_server_B_params(mutated)
    updated = model.get_server_B_params()
    for key in mutated.keys():
        assert torch.allclose(updated[key], mutated[key])

    model.reset_server_B_params()
    reset = model.get_server_B_params()
    for tensor in reset.values():
        assert torch.allclose(tensor, torch.zeros_like(tensor))


def test_dual_b_server_noise_sampling_shapes():
    model = SmallConvNetLoRA(in_channels=1, num_classes=3, input_size=64, hidden_channels=8, lora_r=2)
    server = DualBFedSAServer(model, torch.device("cpu"), {"noise_std": 0.05}, {}, eval_loader=None)
    perturbed = server.sample_perturbed_B(client_id=0)
    assert perturbed, "Perturbed B dict should not be empty"
    for key, tensor in perturbed.items():
        base = server.server_B_params[key]
        assert tensor.shape == base.shape
