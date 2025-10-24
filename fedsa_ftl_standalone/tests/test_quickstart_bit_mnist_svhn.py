import pathlib
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _ensure_opacus_stub():
    if 'opacus' in sys.modules:
        return

    opacus = types.ModuleType("opacus")

    class _DummyPrivacyEngine:
        def __init__(self, *args, **kwargs):
            pass

        def make_private(self, module, optimizer, data_loader, noise_multiplier, max_grad_norm, poisson_sampling=False):
            return module, optimizer, data_loader

    opacus.PrivacyEngine = _DummyPrivacyEngine

    grad_sample = types.ModuleType("opacus.grad_sample")
    utils = types.ModuleType("opacus.grad_sample.utils")

    def _clear_grad_sample(model):
        return None

    utils.clear_grad_sample = _clear_grad_sample
    grad_sample.utils = utils
    grad_sample.clear_grad_sample = _clear_grad_sample

    sys.modules['opacus'] = opacus
    sys.modules['opacus.grad_sample'] = grad_sample
    sys.modules['opacus.grad_sample.utils'] = utils


_ensure_opacus_stub()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickstart_bit_mnist_svhn import BiTFedSAFTLClient  # noqa: E402


class DummyLoRAModel(nn.Module):
    def __init__(self, in_features=4, hidden_rank=2, out_features=3):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        nn.init.zeros_(self.base.weight)
        self.lora_A = nn.Parameter(torch.randn(in_features, hidden_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(hidden_rank, out_features) * 0.01)

    def forward(self, x):
        lora_update = x @ self.lora_A @ self.lora_B
        return self.base(x) + lora_update

    def get_A_parameter_groups(self):
        return [self.lora_A]

    def get_B_parameter_groups(self):
        return [self.lora_B]

    def get_A_parameters(self):
        return {'lora_A': self.lora_A.data.clone()}

    def set_A_parameters(self, params):
        if 'lora_A' in params:
            self.lora_A.data.copy_(params['lora_A'].to(self.lora_A.device, dtype=self.lora_A.dtype))


def _make_client_config(gradient_enabled: bool):
    return {
        'privacy': {'enable_privacy': False},
        'use_amp': False,
        'training': {'epochs': 1, 'lr': 0.1, 'momentum': 0.0, 'weight_decay': 0.0},
        'federated': {'aggregation_method': 'fedsa'},
        'data': {'batch_size': 2, 'num_workers': 0},
        'model': {'num_classes': 3},
        'privacy_attacks': {
            'gradient_leakage': {'enabled': gradient_enabled},
            'membership_inference': {'enabled': False},
        }
    }


def _make_dataloader():
    x = torch.tensor([[0.5, -1.0, 0.3, 1.2],
                      [0.1, 0.7, -0.2, -0.5]], dtype=torch.float32)
    y = torch.tensor([1, 2], dtype=torch.long)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_gradient_payload_generated_when_attack_enabled():
    device = torch.device('cpu')
    model = DummyLoRAModel()
    config = _make_client_config(gradient_enabled=True)

    client = BiTFedSAFTLClient(0, model, device, config)
    dataloader = _make_dataloader()
    result = client.train(dataloader, config['training'])

    payload = result.get('gradient_leakage_payload')
    assert payload is not None
    assert 'delta_A_params' in payload
    assert payload['batch_images'].shape == (2, 4)
    assert payload['lr'] == config['training']['lr']


def test_gradient_payload_absent_when_attack_disabled():
    device = torch.device('cpu')
    model = DummyLoRAModel()
    config = _make_client_config(gradient_enabled=False)

    client = BiTFedSAFTLClient(0, model, device, config)
    dataloader = _make_dataloader()
    result = client.train(dataloader, config['training'])

    assert 'gradient_leakage_payload' not in result
