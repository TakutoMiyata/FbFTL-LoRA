import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attack_svd_dlg import dlg_refinement, svd_feature_leakage  # noqa: E402


class TinyLoRAModel(nn.Module):
    def __init__(self, in_features=4, rank=2, num_classes=3):
        super().__init__()
        self.base = nn.Linear(in_features, num_classes, bias=False)
        nn.init.zeros_(self.base.weight)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.05)
        self.lora_B = nn.Parameter(torch.randn(rank, num_classes) * 0.05)

    def forward(self, x):
        lora_update = x @ self.lora_A @ self.lora_B
        return self.base(x) + lora_update

    def get_A_parameter_groups(self):
        return [{'params': [self.lora_A]}]

    def get_B_parameter_groups(self):
        return [{'params': [self.lora_B]}]

    def get_A_parameters(self):
        return {'lora_A': self.lora_A.detach().clone()}

    def set_A_parameters(self, params):
        if 'lora_A' in params:
            self.lora_A.data.copy_(params['lora_A'])


def _make_delta_from_training_step(model, data, labels, lr=0.5):
    """Run one SGD step to obtain a ΔA payload."""
    base_A = {name: tensor.clone().detach() for name, tensor in model.get_A_parameters().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    logits = model(data)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    updated_A = model.get_A_parameters()
    delta = {name: updated_A[name] - base_A[name] for name in base_A}
    # Restore original weights so tests remain deterministic
    model.set_A_parameters(base_A)
    return delta


def test_svd_feature_leakage_recovers_top_direction():
    torch.manual_seed(0)
    base_delta = torch.tensor(
        [[2.0, 0.0],
         [2.0, 0.0],
         [0.0, 1.0]],
        dtype=torch.float32,
    )
    results, feature = svd_feature_leakage({'layer': base_delta}, topk=1, device=torch.device('cpu'))
    assert feature.shape[0] == base_delta.shape[1]
    # The singular spectrum should prioritize the first column
    assert results.singular_values[0] > 1.0
    top_vec = results.feature_vectors[0] / results.feature_vectors[0].norm()
    assert torch.allclose(top_vec.abs(), torch.tensor([1.0, 0.0]), atol=1e-4)


def test_dlg_refinement_runs_with_small_steps():
    torch.manual_seed(42)
    model = TinyLoRAModel()
    data = torch.randn(2, 4)
    labels = torch.tensor([0, 1], dtype=torch.long)
    delta = _make_delta_from_training_step(model, data, labels, lr=0.3)

    attack_cfg = {
        'attack_lr': 0.5,
        'attack_steps': 10,
        'data_reg': 1e-4,
        'tv_reg': 0.0,
        'grad_weight': 1.0,
        'label_weight': 0.0,
        'optimize_labels': False,
        'num_classes': 3,
    }
    result, recon_images, recon_labels = dlg_refinement(
        model,
        delta,
        batch_images=data,
        batch_labels=labels,
        attack_config=attack_cfg,
        device=torch.device('cpu'),
    )

    assert recon_images.shape == data.shape
    assert recon_labels.shape[0] == labels.shape[0]
    assert result.reconstruction_mse >= 0.0
    assert torch.isfinite(torch.tensor(result.reconstruction_mse))
