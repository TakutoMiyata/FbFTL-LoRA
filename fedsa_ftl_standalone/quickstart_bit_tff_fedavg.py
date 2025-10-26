#!/usr/bin/env python3
"""
Quick start script for BiT (Big Transfer) federated transfer learning on TFF CIFAR-100 with FedAvg
Uses TensorFlow Federated's hierarchical LDA non-IID dataset
FedAvg: Shares ALL trainable parameters (LoRA A + B + classifier)
"""

import torch
import argparse
import yaml
from pathlib import Path
import os
import sys
import numpy as np
import random
import json
import time
from datetime import datetime

# Set GPU device BEFORE any CUDA operations
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Configure TensorFlow to not hog all GPU memory
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Disable tqdm globally BEFORE importing
os.environ['TQDM_DISABLE'] = '1'

# TensorFlow is now optional (not used in this version)
# Using standard CIFAR-100 instead of TFF
TF_AVAILABLE = False
try:
    import tensorflow as tf
    # Limit TensorFlow to only 2GB of GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
            print(f"✅ TensorFlow GPU memory limited to 2GB, growth enabled")
        except RuntimeError as e:
            print(f"⚠️  TensorFlow GPU configuration error: {e}")
    TF_AVAILABLE = True
except ImportError:
    print("ℹ️  TensorFlow not available - using standard CIFAR-100 only")

from tqdm import tqdm

# Opacus for differential privacy
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
    try:
        from opacus.grad_sample.utils import clear_grad_sample
    except ImportError:
        try:
            from opacus.grad_sample import clear_grad_sample
        except ImportError:
            clear_grad_sample = None
except ImportError:
    OPACUS_AVAILABLE = False
    clear_grad_sample = None
    print("Warning: Opacus not available. Install with: pip install opacus")


def manual_clear_grad_sample(model):
    """Manually clear grad_sample attributes from model parameters."""
    cleared_count = 0
    for p in model.parameters():
        if hasattr(p, 'grad_sample'):
            delattr(p, 'grad_sample')
            cleared_count += 1
    if cleared_count > 0:
        print(f"🧹 Manually cleared grad_sample from {cleared_count} parameters")


def safe_clear_grad_sample(model):
    """Safely clear grad_sample attributes using Opacus function or manual fallback."""
    try:
        if clear_grad_sample is not None:
            clear_grad_sample(model)
        else:
            manual_clear_grad_sample(model)
    except Exception:
        try:
            manual_clear_grad_sample(model)
        except Exception:
            pass


def comprehensive_grad_sample_cleanup(model, verbose=False):
    """Comprehensive cleanup of grad_sample attributes and related Opacus artifacts."""
    cleared_params = 0
    cleared_buffers = 0

    for name, param in model.named_parameters():
        if hasattr(param, 'grad_sample'):
            delattr(param, 'grad_sample')
            cleared_params += 1
            if verbose:
                print(f"  Cleared grad_sample from parameter: {name}")

        opacus_attrs = ['_forward_hooks', '_backward_hooks', '_grad_sample', 'summed_grad']
        for attr in opacus_attrs:
            if hasattr(param, attr):
                try:
                    delattr(param, attr)
                    if verbose:
                        print(f"  Cleared {attr} from parameter: {name}")
                except:
                    pass

    for name, buffer in model.named_buffers():
        if hasattr(buffer, 'grad_sample'):
            delattr(buffer, 'grad_sample')
            cleared_buffers += 1
            if verbose:
                print(f"  Cleared grad_sample from buffer: {name}")

    import gc
    gc.collect()

    if verbose and (cleared_params > 0 or cleared_buffers > 0):
        print(f"🧹 Comprehensive cleanup: {cleared_params} params, {cleared_buffers} buffers")


# Load environment variables from .env file
def load_env_file(env_path='.env'):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        print(f"✅ Environment variables loaded from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")

# Load .env file at startup
load_env_file()

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import from src modules
from backbones_bit import make_bit_model_with_lora, print_bit_model_summary
from fedsa_ftl_client import FedSAFTLClient
from fedsa_ftl_server import FedSAFTLServer
# TFF imports removed - using standard CIFAR-100
from privacy_utils import DifferentialPrivacy
from notification_utils import SlackNotifier
from dp_utils import WeightedFedAvg
import torch.nn.functional as F


class BiTFedAvgClient(FedSAFTLClient):
    """Extended client for BiT models with FedAvg (shares all trainable parameters)"""

    def __init__(self, client_id, model, device, config, privacy_mechanism=None):
        super().__init__(client_id, model, device)
        self.config = config
        self.use_dp = config.get('privacy', {}).get('enable_privacy', False)
        self.use_amp = False if self.use_dp else config.get('use_amp', True)
        self.aggregation_method = config.get('federated', {}).get('aggregation_method', 'fedavg')

        self.core_model = model

        training_config = config.get('training', {})
        opt_kwargs = dict(
            lr=training_config.get('lr', 0.001),
            momentum=training_config.get('momentum', 0.9),
            weight_decay=training_config.get('weight_decay', 0.0001)
        )

        if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp':
            self.A_params = list(model.get_A_parameter_groups())
            self.B_params = list(model.get_B_parameter_groups())
            cls_params = []
            if hasattr(model, 'head'):
                cls_params = list(model.head.parameters())
            elif hasattr(model, 'classifier'):
                cls_params = list(model.classifier.parameters())
            elif hasattr(model, 'fc'):
                cls_params = list(model.fc.parameters())

            self.local_params = self.B_params + cls_params

            if len(self.local_params) == 0:
                print(f"Warning: client {client_id} has no local_params (B+classifier).")

            self.dp_optimizer = torch.optim.SGD(self.A_params, **opt_kwargs)
            self.local_optimizer = torch.optim.SGD(self.local_params, **opt_kwargs)

            if OPACUS_AVAILABLE:
                self.privacy_engine = PrivacyEngine()
                self.dp_noise_multiplier = config.get('privacy', {}).get('noise_multiplier', 1.0)
                self.dp_max_grad_norm = config.get('privacy', {}).get('max_grad_norm', 1.0)
                self.privacy_engine_attached = False
                print(f"Client {client_id}: Opacus initialized, will attach during training")
            else:
                raise RuntimeError("Opacus not available")

            self.optimizer = None
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            num_trainable = sum(p.numel() for p in trainable_params)
            print(f"Client {client_id}: Single optimizer tracking {len(trainable_params)} parameter tensors ({num_trainable:,} parameters)")

            self.optimizer = torch.optim.SGD(trainable_params, **opt_kwargs)

            self.dp_optimizer = None
            self.local_optimizer = None
            self.privacy_engine = None
            self.privacy_engine_attached = False

        self.local_data_size = 0

        if privacy_mechanism is not None:
            self.set_privacy_mechanism(privacy_mechanism)

        comprehensive_grad_sample_cleanup(self.model, verbose=False)

    def _unwrap(self):
        """Unwrap Opacus GradSampleModule to access original model methods"""
        return getattr(self.model, "_module", self.model)

    def train(self, dataloader, training_config):
        """Train the model with A-only DP-SGD (via Opacus) and non-DP B+classifier"""
        self.model.train()
        self.local_data_size = len(dataloader.dataset)

        if (self.use_dp and self.aggregation_method == 'fedsa_shareA_dp' and
            hasattr(self, 'privacy_engine') and not self.privacy_engine_attached):
            self.model, self.dp_optimizer, _ = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.dp_optimizer,
                data_loader=dataloader,
                noise_multiplier=self.dp_noise_multiplier,
                max_grad_norm=self.dp_max_grad_norm,
                poisson_sampling=False
            )
            self.privacy_engine_attached = True
            print(f"Client {self.client_id}: Opacus attached to model with A-only optimizer")

        total_loss = 0.0
        correct = 0
        total = 0
        num_epochs = training_config.get('epochs', 5)

        model_is_half = next(self.model.parameters()).dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler() if (self.use_amp and torch.cuda.is_available() and not model_is_half) else None

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                if model_is_half:
                    data = data.half()

                if len(target.shape) > 1:
                    target_for_loss = target
                    target_for_acc = target.argmax(dim=1)
                else:
                    target_for_loss = target
                    target_for_acc = target

                if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp' and self.privacy_engine_attached:
                    for p in self.model.parameters():
                        if not hasattr(p, 'grad_sample'):
                            p.grad_sample = None

                    self.dp_optimizer.zero_grad()
                    self.local_optimizer.zero_grad()

                    output = self.model(data)
                    if len(target.shape) > 1:
                        loss = -(target_for_loss * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, target_for_loss)

                    loss.backward()

                    self.dp_optimizer.step()
                    self.local_optimizer.step()

                elif self.aggregation_method == 'fedsa_shareA_dp':
                    self.dp_optimizer.zero_grad()
                    self.local_optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.use_amp and scaler is not None):
                        output = self.model(data)
                    if len(target.shape) > 1:
                        loss = -(target_for_loss * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, target_for_loss)

                    loss.backward()
                    self.dp_optimizer.step()
                    self.local_optimizer.step()

                else:
                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.use_amp and scaler is not None):
                        output = self.model(data)
                    if len(target.shape) > 1:
                        loss = -(target_for_loss * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, target_for_loss)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_for_acc.view_as(pred)).sum().item()
                total += target_for_acc.size(0)

            total_loss += epoch_loss

            if torch.cuda.is_available():
                if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp':
                    torch.cuda.empty_cache()
                elif epoch % 2 == 1:
                    torch.cuda.empty_cache()

        avg_loss = total_loss / (num_epochs * len(dataloader))
        accuracy = 100. * correct / total

        if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp':
            self.dp_optimizer.zero_grad()

            for p in self.model.parameters():
                if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                    p.grad_sample = None
        else:
            safe_clear_grad_sample(self.model)

        # FedAvg: Return ALL trainable parameters (not just A matrices)
        all_params = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}

        update = {
            'model_params': all_params,  # All trainable params (A + B + classifier)
            'num_samples': self.local_data_size,
            'local_data_size': self.local_data_size,
            'loss': avg_loss,
            'accuracy': accuracy,
            'upload_type': 'all_trainable_parameters'
        }

        if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp' and self.privacy_engine_attached:
            delta = self.config.get('privacy', {}).get('delta', 1e-5)
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=delta)
            update['privacy_analysis'] = {
                'epsilon': epsilon,
                'delta': delta,
                'noise_multiplier': self.config['privacy'].get('noise_multiplier', 1.0),
                'max_grad_norm': self.config['privacy'].get('max_grad_norm', 1.0),
                'batch_size': self.config['data'].get('batch_size', 128),
                'dataset_size': self.local_data_size,
                'method': 'Opacus DP-SGD (A-only)'
            }

        return update

    def update_model(self, global_params):
        """Update model with global parameters (FedAvg)"""
        if 'model_params' in global_params:
            # Update ALL trainable parameters
            for name, param in self.model.named_parameters():
                if name in global_params['model_params'] and param.requires_grad:
                    param.data.copy_(global_params['model_params'][name])
            print(f"Client {self.client_id}: Updated all trainable parameters from server")

    def reset_A_optimizer_state(self):
        """Reset optimizer state for A parameters after server update"""
        pass

    def get_model_size(self):
        """Get model size information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self.aggregation_method == 'fedsa_shareA_dp':
            A_params = sum(p.numel() for p in self._unwrap().get_A_parameter_groups())
            B_params = sum(p.numel() for p in self._unwrap().get_B_parameter_groups())
            communication_params = A_params
            compression_ratio = total_params / communication_params if communication_params > 0 else 1.0
        else:
            communication_params = trainable_params
            compression_ratio = total_params / trainable_params if trainable_params > 0 else 1.0

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'communication_params': communication_params,
            'compression_ratio': compression_ratio
        }


def main():
    slack_notifier = None
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if webhook_url:
        slack_notifier = SlackNotifier(webhook_url)
        print("Slack notifications enabled")
    else:
        print("Slack notifications disabled")

    parser = argparse.ArgumentParser(description='FedSA-LoRA BiT with TFF CIFAR-100 (A matrices only)')
    parser.add_argument('--config', type=str, default='configs/experiment_configs_non_iid/bit_tff_cifar100.yaml',
                       help='Path to configuration file')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Override number of rounds')
    parser.add_argument('--clients', type=int, default=None,
                       help='Override number of clients')
    parser.add_argument('--model', type=str,
                       choices=['bit_s_r50x1', 'bit_m_r50x1', 'bit_s_r101x1', 'bit_m_r101x1'],
                       default=None, help='Override BiT model variant')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Creating default configuration for BiT on TFF CIFAR-100...")

        default_config = {
            'seed': 42,
            'use_gpu': True,
            'data': {
                'dataset_name': 'tff_cifar100',
                'data_dir': './data',
                'num_clients': 10,
                'num_test_clients': 30,
                'batch_size': 64,
                'num_workers': 4,
                'input_size': 224,
                'verbose': False,
                'augmentations': {
                    'horizontal_flip': {'enabled': True, 'prob': 0.5},
                    'random_rotation': {'enabled': True, 'degrees': 10},
                    'random_resized_crop': {'enabled': True, 'scale_min': 0.5},
                }
            },
            'model': {
                'model_name': 'bit_m_r50x1',
                'num_classes': 100,
                'pretrained': True,
                'freeze_backbone': True,
                'lora': {
                    'enabled': True,
                    'r': 8,
                    'alpha': 16,
                    'dropout': 0.1
                }
            },
            'training': {
                'epochs': 5,
                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'scheduler': 'cosine'
            },
            'federated': {
                'num_rounds': 200,
                'num_clients': 10,
                'client_fraction': 1.0,
                'aggregation_method': 'fedsa',
                'checkpoint_freq': 25
            },
            'privacy': {
                'enable_privacy': False,
                'epsilon': 10.0,
                'delta': 1e-5,
                'max_grad_norm': 0.5
            },
            'evaluation': {
                'eval_freq': 25
            },
            'experiment': {
                'name': 'BiT_TFF_CIFAR100_NonIID',
                'output_dir': 'experiments/quickstart_bit_tff'
            },
            'reproducibility': {
                'deterministic': False
            }
        }

        configs_dir = Path("configs/experiment_configs_non_iid")
        configs_dir.mkdir(parents=True, exist_ok=True)
        config_path = configs_dir / "bit_tff_cifar100.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        print(f"Default configuration saved to: {config_path}")
        config = default_config
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    if args.rounds:
        config['federated']['num_rounds'] = args.rounds
    if args.clients:
        config['federated']['num_clients'] = args.clients
        config['data']['num_clients'] = args.clients
    if args.model:
        config['model']['model_name'] = args.model

    device = torch.device('cuda' if config.get('use_gpu', False) and torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("FedSA-LoRA BiT Federated Learning with TFF CIFAR-100 (A matrices only)")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Dataset: TFF CIFAR-100 (hierarchical LDA non-IID)")
    print(f"Training Clients: {config['federated']['num_clients']}")
    print(f"Test Clients: {config['data'].get('num_test_clients', 30)}")
    print(f"Rounds: {config['federated']['num_rounds']}")
    privacy_enabled = config.get('privacy', {}).get('enable_privacy', False)
    print(f"Privacy: {'Enabled' if privacy_enabled else 'Disabled'}")
    if privacy_enabled:
        agg_method = config['federated'].get('aggregation_method', 'fedsa')
        if agg_method == 'fedsa_shareA_dp':
            print("Privacy Method: Opacus DP-SGD (A-only, RDP accountant)")
    print("=" * 80)

    if 'seed' in config:
        seed = config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if config.get('reproducibility', {}).get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    if config.get('privacy', {}).get('enable_privacy', False):
        if not OPACUS_AVAILABLE:
            raise RuntimeError("Privacy is enabled but Opacus is not installed.")
        print("✅ Opacus is available for Differential Privacy")

    # Use standard CIFAR-100 with data_utils (both IID and non-IID)
    print("\nPreparing standard CIFAR-100 federated data...")
    from data_utils import prepare_federated_data, get_client_dataloader

    trainset, testset, client_train_indices, client_test_indices = prepare_federated_data(config['data'])

    data_split = config['data'].get('data_split', 'iid')
    print(f"\n✅ Standard CIFAR-100 data loaded:")
    print(f"  Training clients: {len(client_train_indices)}")
    print(f"  Test clients: {len(client_test_indices)}")
    print(f"  Split method: {data_split.upper()}")
    print(f"  Using standard torchvision CIFAR-100")

    # Convert to uniform format for compatibility
    train_datasets = {}
    test_datasets = {}
    for i, indices in enumerate(client_train_indices):
        train_datasets[i] = (trainset, indices)
    for i, indices in enumerate(client_test_indices):
        test_datasets[i] = (testset, indices)

    client_info = {
        'num_train_clients': len(client_train_indices),
        'num_test_clients': len(client_test_indices),
        'split_method': f'{data_split.upper()} (standard CIFAR-100)',
        'use_standard_cifar': True
    }

    print(f"\nCreating BiT model...")
    initial_model = make_bit_model_with_lora(config)
    initial_model = initial_model.to(device)
    print_bit_model_summary(config, initial_model)

    total_params = sum(p.numel() for p in initial_model.parameters())
    trainable_params = sum(p.numel() for p in initial_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"🔍 Transfer Learning Check:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    if trainable_params / total_params > 0.5:
        print(f"  ⚠️  WARNING: Too many trainable params!")

    print("Initializing federated server...")
    server = FedSAFTLServer(device)
    # FedAvg: Initialize with all trainable parameters
    server.global_params = {
        'model_params': {name: param.data.clone() for name, param in initial_model.named_parameters() if param.requires_grad}
    }
    print(f"✅ Server initialized with all trainable parameters")

    temp_client = BiTFedAvgClient(0, initial_model, device, config)
    model_stats = temp_client.get_model_size()

    print("\nModel Statistics:")
    print(f"  Architecture: {config['model']['model_name']}")
    print(f"  Total parameters: {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    lora_rank = config['model'].get('lora', {}).get('r', 0)
    if lora_rank > 0:
        print(f"  LoRA rank: {lora_rank}")
    print(f"  Communication overhead: {model_stats['communication_params']:,} parameters/round")
    print(f"  Compression ratio: {model_stats['compression_ratio']:.2f}x")

    print(f"\nCreating {config['federated']['num_clients']} BiT federated clients...")
    clients = []

    current_date = datetime.now().strftime('%m%d')
    base_experiment_dir = Path(config.get('experiment', {}).get('output_dir', 'experiments/quickstart_bit_tff'))
    agg_method = config.get('federated', {}).get('aggregation_method', 'unknown')
    method_subdir = agg_method.lower()
    method_dir = base_experiment_dir / method_subdir
    experiment_dir = method_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {experiment_dir}")

    date_bit_suffix = f"{current_date}BiT"

    results = {
        'config': config,
        'start_time': datetime.now().isoformat(),
        'experiment_dir': str(experiment_dir),
        'date_suffix': date_bit_suffix,
        'rounds': [],
        'summary': {}
    }

    if slack_notifier:
        config_for_notification = {
            'experiment': {'name': f'BiT TFF QuickStart ({config["model"]["model_name"]})'},
            'federated': config['federated'],
            'privacy': config.get('privacy', {})
        }
        slack_notifier.send_training_start(config_for_notification)

    privacy_enabled = config.get('privacy', {}).get('enable_privacy', False)

    # Use simple integer indices for all clients
    train_client_ids = list(range(client_info['num_train_clients']))
    test_client_ids = list(range(client_info['num_test_clients']))

    for i, client_id in enumerate(train_client_ids):
        privacy_mechanism = None
        if privacy_enabled:
            privacy_mechanism = DifferentialPrivacy(
                epsilon=config['privacy'].get('epsilon', 10.0),
                delta=config['privacy'].get('delta', 1e-5),
                max_grad_norm=config['privacy'].get('max_grad_norm', 0.5),
                total_rounds=config['federated'].get('num_rounds', 200)
            )

        client_model = make_bit_model_with_lora(config)
        client_model = client_model.to(device)

        # FedAvg: Synchronize with server's all trainable parameters
        if hasattr(server, 'global_params') and server.global_params:
            for name, param in client_model.named_parameters():
                if name in server.global_params['model_params'] and param.requires_grad:
                    param.data.copy_(server.global_params['model_params'][name])

        client = BiTFedAvgClient(
            i,
            client_model,
            device,
            config,
            privacy_mechanism
        )
        clients.append(client)

    print(f"✅ All {len(clients)} clients initialized")
    print("Starting BiT FedAvg training with TFF CIFAR-100...")
    print("=" * 80)

    best_accuracy = 0
    best_round = 0
    start_time = time.time()

    round_pbar = tqdm(range(config['federated']['num_rounds']),
                     desc="Federated Rounds",
                     unit="round")

    for round_idx in round_pbar:
        print(f"\n[Round {round_idx + 1}/{config['federated']['num_rounds']}]")
        round_start_time = time.time()

        client_fraction = config['federated'].get('client_fraction', 1.0)
        num_clients = config['federated']['num_clients']
        num_selected = max(1, int(np.ceil(client_fraction * num_clients)))

        if client_fraction >= 1.0:
            selected_clients = list(range(num_clients))
        else:
            selected_clients = sorted(random.sample(range(num_clients), num_selected))

        print(f"Selected clients: {selected_clients}")

        client_updates = []
        train_accuracies = []

        client_pbar = tqdm(selected_clients,
                          desc="Training clients",
                          leave=False,
                          unit="client")

        for client_idx in client_pbar:
            client_pbar.set_description(f"Training client {client_idx}")

            # Use standard CIFAR-100 dataloader
            trainset, indices = train_datasets[client_idx]
            client_dataloader = get_client_dataloader(
                trainset,
                indices,
                batch_size=config['data']['batch_size'],
                shuffle=True,
                num_workers=config['data'].get('num_workers', 0)
            )

            # FedAvg: Update with all global parameters
            if hasattr(server, 'global_params') and server.global_params:
                clients[client_idx].update_model(server.global_params)

            client_result = clients[client_idx].train(client_dataloader, config['training'])
            client_updates.append(client_result)
            train_accuracies.append(client_result['accuracy'])

            print(f"  Client {client_idx}: Loss={client_result['loss']:.4f}, "
                  f"Accuracy={client_result['accuracy']:.2f}%")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        is_eval_round = (round_idx + 1) % config.get('evaluation', {}).get('eval_freq', 25) == 0

        if is_eval_round:
            print("\nEvaluating personalized models (each training client on fixed test set)...")

            # FedSA-LoRA: No global model exists (only A matrices are shared)
            # Each training client has personalized model (A_shared + B_local + classifier_local)
            # Evaluate each training client's personalized model on fixed test set

            personalized_results = []
            personalized_accuracies = []

            # Create combined test dataloader from all test clients
            # Combine all test client indices
            all_test_indices = []
            for test_idx in range(len(test_datasets)):
                test_set_ref, indices = test_datasets[test_idx]
                all_test_indices.extend(indices)

            # Use the testset reference from first client
            test_set_ref, _ = test_datasets[0]
            combined_test_loader = get_client_dataloader(
                test_set_ref,
                all_test_indices,
                batch_size=config['data']['batch_size'],
                shuffle=False,
                num_workers=config['data'].get('num_workers', 0)
            )
            print(f"  Combined test set: {len(all_test_indices)} samples from {len(test_datasets)} test clients")

            for client_idx in selected_clients:
                # Evaluate this training client's personalized model on combined test set
                test_result = clients[client_idx].evaluate(combined_test_loader)
                personalized_results.append(test_result)
                personalized_accuracies.append(test_result['accuracy'])
                print(f"    Training Client {client_idx} (personalized): {test_result['accuracy']:.2f}%")

            avg_personalized_acc = sum(personalized_accuracies) / len(personalized_accuracies)
            print(f"  Average Personalized Accuracy: {avg_personalized_acc:.2f}%")

            client_test_results = personalized_results
            test_accuracies = personalized_accuracies
        else:
            test_accuracies = None
            personalized_accuracies = None
            avg_personalized_acc = None
            client_test_results = [{'accuracy': 0, 'loss': 0} for _ in selected_clients]

        aggregation_method = config['federated'].get('aggregation_method', 'fedavg')

        if aggregation_method != 'fedavg':
            raise ValueError(f"This script only supports FedAvg. Use quickstart_bit_tff.py for FedSA.")

        # FedAvg: Aggregate ALL trainable parameters
        client_model_params = [update['model_params'] for update in client_updates]
        client_sample_counts = [update['local_data_size'] for update in client_updates]

        # Weighted average of all parameters
        aggregated_params = {}
        total_samples = sum(client_sample_counts)

        for param_name in client_model_params[0].keys():
            weighted_sum = sum(
                client_params[param_name] * (count / total_samples)
                for client_params, count in zip(client_model_params, client_sample_counts)
            )
            aggregated_params[param_name] = weighted_sum

        if not hasattr(server, 'global_params'):
            server.global_params = {}
        server.global_params = {'model_params': aggregated_params}

        print(f"=== FedAvg Aggregation ===")
        print(f"Total clients: {len(client_updates)}")
        print(f"Total samples: {total_samples}")
        print(f"Parameters aggregated: {len(aggregated_params)}")
        print(f"=== ALL trainable params shared (A + B + classifier) ===")

        if round_idx % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            import gc
            gc.collect()

            for client in clients:
                comprehensive_grad_sample_cleanup(client.model, verbose=False)

        def get_bytes_per_param(tensor):
            if tensor.dtype in (torch.float16, torch.bfloat16):
                return 2
            else:
                return 4

        # FedAvg: Calculate communication cost for ALL trainable parameters
        total_bytes = sum(p.numel() * get_bytes_per_param(p) for p in aggregated_params.values())
        communication_cost_mb = total_bytes / (1024 * 1024)

        total_params = sum(p.numel() for p in aggregated_params.values())

        for i, update in enumerate(client_updates):
            if 'upload_type' in update and update['upload_type'] != 'all_trainable_parameters':
                print(f"WARNING: Client {i} uploaded unexpected parameters!")

        current_epsilon = None
        if config.get('privacy', {}).get('enable_privacy', False):
            privacy_analyses = [update.get('privacy_analysis', {}) for update in client_updates if 'privacy_analysis' in update]
            if privacy_analyses:
                epsilons = [analysis.get('epsilon', 0) for analysis in privacy_analyses if 'epsilon' in analysis]

                if epsilons:
                    avg_eps = sum(epsilons) / len(epsilons)
                    max_eps = max(epsilons)
                    current_epsilon = max_eps

                    print(f"\n  === Privacy Analysis (Opacus DP-SGD A-only) ===")
                    print(f"  Current ε (avg): {avg_eps:.4f}, ε (max): {max_eps:.4f}")
                    print(f"  Privacy Budget: ε≤{config['privacy'].get('epsilon', 8.0)}")
                    print(f"  δ: {config['privacy'].get('delta', 1e-5)}")
                    print(f"  =============================================")

        round_stats = {
            'communication_cost_mb': communication_cost_mb,
            'aggregated_params': total_params,
            'aggregation_method': aggregation_method
        }

        round_time = time.time() - round_start_time
        total_time = time.time() - start_time
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)

        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Avg Training Accuracy: {avg_train_acc:.2f}%")

        is_new_best = False
        if is_eval_round and test_accuracies is not None:
            print(f"  Avg Test Accuracy (FedAvg global model): {avg_personalized_acc:.2f}%")
            if avg_personalized_acc > best_accuracy:
                best_accuracy = avg_personalized_acc
                best_round = round_idx + 1
                is_new_best = True
                print(f"  ** New best test accuracy! **")

        print(f"  Communication Cost (per-round): {round_stats.get('communication_cost_mb', 0):.2f} MB")
        print(f"  Round time: {round_time/60:.1f} min ({round_time:.0f}s)")
        print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f}h)")

        rounds_completed = round_idx + 1
        if rounds_completed > 0:
            avg_round_time = total_time / rounds_completed
            remaining_rounds = config['federated']['num_rounds'] - rounds_completed
            estimated_remaining = avg_round_time * remaining_rounds
            print(f"  Estimated remaining time: {estimated_remaining/3600:.1f}h ({estimated_remaining/60:.0f} min)")

        round_pbar.set_postfix({
            'train_acc': f'{avg_train_acc:.2f}%',
            'test_acc': f'{avg_personalized_acc:.2f}%' if avg_personalized_acc else 'N/A',
            'best': f'{best_accuracy:.2f}%',
            'time': f'{round_time:.1f}s'
        })

        round_result = {
            'round': round_idx + 1,
            'timestamp': datetime.now().isoformat(),
            'selected_clients': selected_clients,
            'avg_train_accuracy': avg_train_acc,
            'avg_personalized_accuracy': avg_personalized_acc if avg_personalized_acc is not None else None,
            'individual_train_accuracies': train_accuracies,
            'individual_test_accuracies': test_accuracies if test_accuracies is not None else None,
            'communication_cost_mb': round_stats.get('communication_cost_mb', 0),
            'is_best_round': is_new_best
        }

        if current_epsilon is not None:
            round_result['epsilon'] = current_epsilon

        results['rounds'].append(round_result)

        results_file = experiment_dir / f'training_results_{date_bit_suffix}.json'

        results_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save results file: {e}")

        if slack_notifier and (round_idx + 1) % 25 == 0:
            config_for_notification = {
                'experiment': {'name': f'BiT TFF QuickStart ({config["model"]["model_name"]})'},
                'federated': config['federated']
            }

            round_stats_for_notification = {
                'train_accuracy': avg_train_acc,
                'test_accuracy': avg_personalized_acc if avg_personalized_acc is not None else None,
                'per_round_communication_mb': round_stats.get('communication_cost_mb', 0)
            }

            server_summary_for_notification = {
                'best_test_accuracy': best_accuracy
            }

            slack_notifier.send_progress_update(
                config_for_notification,
                round_idx,
                round_stats_for_notification,
                server_summary_for_notification
            )

        if (round_idx + 1) % config['federated'].get('checkpoint_freq', 25) == 0:
            print(f"  Checkpoint saved at round {round_idx + 1}")
            checkpoint_file = experiment_dir / f'checkpoint_round_{round_idx + 1}_{date_bit_suffix}.json'
            checkpoint_data = {
                'round': round_idx + 1,
                'best_accuracy': best_accuracy,
                'best_round': best_round,
                'current_accuracy': avg_personalized_acc,
                'recent_rounds': results['rounds'][-10:]
            }

            try:
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                print(f"  Checkpoint saved to: {checkpoint_file}")
            except Exception as e:
                print(f"  Warning: Failed to save checkpoint: {e}")

    end_time = time.time()
    training_duration = end_time - start_time

    results['end_time'] = datetime.now().isoformat()
    results['training_duration_seconds'] = training_duration
    total_comm_mb = sum(r['communication_cost_mb'] for r in results['rounds'])
    avg_per_round_comm_mb = total_comm_mb / len(results['rounds']) if results['rounds'] else 0

    results['summary'] = {
        'best_test_accuracy': best_accuracy,
        'best_round': best_round,
        'total_rounds': config['federated']['num_rounds'],
        'total_communication_mb': total_comm_mb,
        'avg_per_round_communication_mb': avg_per_round_comm_mb,
        'final_avg_accuracy': avg_personalized_acc if 'avg_personalized_acc' in locals() and avg_personalized_acc is not None else None,
        'training_duration_hours': training_duration / 3600,
        'model_name': config['model']['model_name'],
        'dataset': 'TFF_CIFAR100'
    }

    final_results_file = experiment_dir / f'final_results_{date_bit_suffix}.json'

    try:
        final_results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {final_results_file}")
    except Exception as e:
        print(f"Warning: Failed to save final results: {e}")

    try:
        import pandas as pd
        df_data = []
        for round_data in results['rounds']:
            row_data = {
                'round': round_data['round'],
                'train_accuracy': round_data['avg_train_accuracy'],
                'test_accuracy': round_data['avg_personalized_accuracy'],
                'per_round_communication_mb': round_data['communication_cost_mb'],
                'is_best': round_data['is_best_round']
            }
            if 'epsilon' in round_data:
                row_data['epsilon'] = round_data['epsilon']
            df_data.append(row_data)

        if df_data:
            df = pd.DataFrame(df_data)
            csv_file = experiment_dir / f'results_summary_{date_bit_suffix}.csv'
            df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved to: {csv_file}")
        else:
            csv_file = None

    except ImportError:
        print("Warning: pandas not available, skipping CSV export")
        csv_file = None
    except Exception as e:
        print(f"Warning: Failed to create CSV summary: {e}")
        csv_file = None

    print("\n" + "=" * 80)
    print("BiT Federated Learning Complete!")
    print("=" * 80)
    print(f"Configuration: {config_path.name}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Best Personalized Accuracy: {best_accuracy:.2f}% (Round {best_round})")
    print(f"Total Rounds: {config['federated']['num_rounds']}")
    total_comm_mb = sum(r['communication_cost_mb'] for r in results['rounds'])
    avg_per_round_mb = total_comm_mb / len(results['rounds']) if results['rounds'] else 0
    print(f"Total Communication: {total_comm_mb:.2f} MB")
    print(f"Average per-round: {avg_per_round_mb:.2f} MB/round")
    print(f"Training Duration: {training_duration / 3600:.2f} hours")
    print(f"Results saved to: {experiment_dir}")
    print(f"  - Training results: final_results_{date_bit_suffix}.json")
    if csv_file:
        print(f"  - Summary CSV: results_summary_{date_bit_suffix}.csv")
    print(f"  - File naming pattern: *_{date_bit_suffix}.*")
    print("=" * 80)

    if slack_notifier:
        config_for_notification = {
            'experiment': {
                'name': f'BiT TFF QuickStart ({config["model"]["model_name"]})',
                'output_dir': str(experiment_dir)
            },
            'federated': config['federated'],
            'privacy': config.get('privacy', {})
        }

        total_comm_final = sum(r['communication_cost_mb'] for r in results['rounds'])
        avg_per_round_final = total_comm_final / len(results['rounds']) if results['rounds'] else 0

        summary_for_notification = {
            'final_avg_accuracy': avg_personalized_acc if 'avg_personalized_acc' in locals() else 0,
            'final_std_accuracy': 0,
            'best_test_accuracy': best_accuracy,
            'total_rounds': config['federated']['num_rounds'],
            'total_communication_mb': total_comm_final,
            'avg_per_round_communication_mb': avg_per_round_final
        }

        slack_notifier.send_training_complete(
            config_for_notification,
            summary_for_notification,
            training_duration
        )

    if best_accuracy > 20:
        print("🎉 BiT federated transfer learning successful!")
    elif best_accuracy > 10:
        print("⚠️ BiT transfer learning in progress - consider more rounds")
    else:
        print("❌ BiT transfer learning may need hyperparameter tuning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        if webhook_url:
            try:
                slack_notifier = SlackNotifier(webhook_url)
                config_for_notification = {
                    'experiment': {'name': 'BiT TFF QuickStart (ERROR)'}
                }
                slack_notifier.send_error_notification(
                    config_for_notification,
                    str(e) + "\n" + traceback.format_exc()[:500]
                )
            except Exception:
                pass
