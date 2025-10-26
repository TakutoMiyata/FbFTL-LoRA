import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from privacy_attacks import (  # noqa: E402
    run_membership_inference_attack,
    run_gradient_leakage_attack,
)


class TinyLoRAModel(nn.Module):
    """Minimal model exposing LoRA-style A adapters for tests."""

    def __init__(self, input_dim=3, hidden_dim=12, num_classes=2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.adapter_A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        hidden = torch.relu(self.encoder(x))
        adapted = hidden + torch.matmul(hidden, self.adapter_A)
        return self.head(adapted)

    def get_A_parameters(self):
        return {'adapter_A': self.adapter_A.detach().clone()}

    def get_A_parameter_groups(self):
        return [self.adapter_A]


def _make_dataloader(features, labels, batch_size):
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_membership_inference_attack_separates_member_and_nonmember():
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    member_features = torch.randn(128, 4)
    member_labels = torch.zeros(128, dtype=torch.long)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        logits = model(member_features.to(device))
        loss = F.cross_entropy(logits, member_labels.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    member_loader = _make_dataloader(member_features, member_labels, batch_size=32)

    nonmember_features = torch.randn(128, 4)
    nonmember_labels = torch.ones(128, dtype=torch.long)
    nonmember_loader = _make_dataloader(nonmember_features, nonmember_labels, batch_size=32)

    result = run_membership_inference_attack(
        model,
        member_loader,
        nonmember_loader,
        device=device,
    )

    assert result.accuracy > 0.7
    assert not math.isnan(result.auc)
    assert result.num_members == 128
    assert result.num_nonmembers == 128
    assert result.method == 'threshold'


def test_shadow_membership_attack_trains_linear_classifier():
    torch.manual_seed(1)

    model = TinyLoRAModel(input_dim=3, hidden_dim=8, num_classes=2)
    device = torch.device('cpu')
    model = model.to(device)

    member_features = torch.randn(256, 3)
    member_labels = torch.zeros(256, dtype=torch.long)
    nonmember_features = torch.randn(256, 3) + 1.0
    nonmember_labels = torch.ones(256, dtype=torch.long)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        logits = model(member_features.to(device))
        loss = F.cross_entropy(logits, member_labels.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    member_loader = _make_dataloader(member_features, member_labels, batch_size=64)
    nonmember_loader = _make_dataloader(nonmember_features, nonmember_labels, batch_size=64)

    result = run_membership_inference_attack(
        model,
        member_loader,
        nonmember_loader,
        device=device,
        method='shadow',
        shadow_config={
            'lr': 0.1,
            'train_steps': 200,
            'batch_size': 128,
            'feature_set': ['delta_l2', 'delta_abs_mean'],
            'feature_update_lr': 0.05,
            'feature_update_steps': 1,
        },
    )

    assert result.accuracy > 0.6
    assert result.method == 'shadow'


def test_gradient_leakage_attack_reconstructs_batch():
    torch.manual_seed(0)

    model = nn.Linear(4, 3)
    device = torch.device('cpu')
    model = model.to(device)
    model.train()

    original_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    data = torch.tensor([[0.5, -1.2, 0.3, 2.0]], dtype=torch.float32)
    labels = torch.tensor([2], dtype=torch.long)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    optimizer.zero_grad()

    output = model(data.to(device))
    loss = F.cross_entropy(output, labels.to(device))
    loss.backward()
    optimizer.step()

    delta_params = {
        'weight': model.weight.detach().clone() - original_params['weight'],
    }

    payload = {
        'initial_A_params': {'weight': original_params['weight']},
        'delta_A_params': delta_params,
        'batch_images': data.clone(),
        'batch_labels': labels.clone(),
        'lr': 0.25,
        'num_classes': 3,
    }

    config = {
        'optimization_steps': 120,
        'attack_lr': 0.4,
        'max_layers': 1,
        'optimize_labels': False,
        'l2_regularizer': 1e-5,
        'max_seconds': 5.0,
        'target_layers': ['weight'],
    }

    result = run_gradient_leakage_attack(
        model,
        payload,
        device=device,
        attack_config=config,
    )

    assert result.reconstruction_mse < 0.02
    assert result.label_accuracy == 1.0
    assert result.cosine_similarity > 0.95
