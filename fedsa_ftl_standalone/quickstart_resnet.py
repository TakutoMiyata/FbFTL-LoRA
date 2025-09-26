#!/usr/bin/env python3
"""
Quick start script for MobileNetV2 federated transfer learning on CIFAR-100
Adapted from quickstart_vit.py for ResNet models
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
import os
# Disable tqdm globally BEFORE importing
os.environ['TQDM_DISABLE'] = '1'


from tqdm import tqdm

# Opacus for differential privacy
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
    # Try to import clear_grad_sample, but don't fail if it doesn't exist
    try:
        from opacus.grad_sample.utils import clear_grad_sample
    except ImportError:
        try:
            from opacus.grad_sample import clear_grad_sample
        except ImportError:
            clear_grad_sample = None  # Function may not exist in newer versions
except ImportError:
    OPACUS_AVAILABLE = False
    clear_grad_sample = None
    print("Warning: Opacus not available. Install with: pip install opacus")


def manual_clear_grad_sample(model):
    """
    Manually clear grad_sample attributes from model parameters.
    This is a fallback when Opacus clear_grad_sample is not available.
    
    Args:
        model: PyTorch model with potential grad_sample attributes
    """
    cleared_count = 0
    for p in model.parameters():
        if hasattr(p, 'grad_sample'):
            delattr(p, 'grad_sample')
            cleared_count += 1
    if cleared_count > 0:
        print(f"🧹 Manually cleared grad_sample from {cleared_count} parameters")


def safe_clear_grad_sample(model):
    """
    Safely clear grad_sample attributes using Opacus function or manual fallback.
    
    Args:
        model: PyTorch model with potential grad_sample attributes
    """
    try:
        # First try the official Opacus method if available
        if clear_grad_sample is not None:
            clear_grad_sample(model)
        else:
            manual_clear_grad_sample(model)
    except Exception as e:
        # Fallback to manual method if official method fails
        try:
            manual_clear_grad_sample(model)
        except Exception as e2:
            # If both fail, just continue (grad_sample may already be cleared)
            pass


def comprehensive_grad_sample_cleanup(model, verbose=False):
    """
    Comprehensive cleanup of grad_sample attributes and related Opacus artifacts.
    
    Args:
        model: PyTorch model
        verbose: Print detailed cleanup information
    """
    cleared_params = 0
    cleared_buffers = 0
    
    # Clear grad_sample from parameters
    for name, param in model.named_parameters():
        if hasattr(param, 'grad_sample'):
            delattr(param, 'grad_sample')
            cleared_params += 1
            if verbose:
                print(f"  Cleared grad_sample from parameter: {name}")
        
        # Also clear other potential Opacus attributes
        opacus_attrs = ['_forward_hooks', '_backward_hooks', '_grad_sample', 'summed_grad']
        for attr in opacus_attrs:
            if hasattr(param, attr):
                try:
                    delattr(param, attr)
                    if verbose:
                        print(f"  Cleared {attr} from parameter: {name}")
                except:
                    pass
    
    # Clear grad_sample from buffers (less common but possible)
    for name, buffer in model.named_buffers():
        if hasattr(buffer, 'grad_sample'):
            delattr(buffer, 'grad_sample')
            cleared_buffers += 1
            if verbose:
                print(f"  Cleared grad_sample from buffer: {name}")
    
    # Force garbage collection after cleanup
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
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        print(f"✅ Environment variables loaded from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")

# Load .env file at startup
load_env_file()

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import from src modules
from cifar_resnet_lora import create_cifar_resnet_lora
from backbones_imagenet import make_model_with_lora, print_model_summary
from fedsa_ftl_client import FedSAFTLClient
from fedsa_ftl_server import FedSAFTLServer
from data_utils import prepare_federated_data, get_client_dataloader, MixupCutmixCollate
from privacy_utils import DifferentialPrivacy
from notification_utils import SlackNotifier
from dp_utils import WeightedFedAvg
import torch.nn.functional as F
import math


class ResNetFedSAFTLClient(FedSAFTLClient):
    """Extended client for ResNet models with DP-LoRA support"""
    
    def __init__(self, client_id, model, device, config, privacy_mechanism=None):
        super().__init__(client_id, model, device)
        self.config = config
        self.use_dp = config.get('privacy', {}).get('enable_privacy', False)
        # Force AMP disabled when DP is enabled
        self.use_amp = False if self.use_dp else config.get('use_amp', True)
        self.aggregation_method = config.get('federated', {}).get('aggregation_method', 'fedavg')
        
        # Store core model reference for safe access after Opacus wrapping
        self.core_model = model
        
        # Get training configuration
        training_config = config.get('training', {})
        opt_kwargs = dict(
            lr=training_config.get('lr', 0.001),
            momentum=training_config.get('momentum', 0.9),
            weight_decay=training_config.get('weight_decay', 0.0001)
        )
        
        if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp':
            # Separate A and B parameters
            self.A_params = list(model.get_A_parameter_groups())
            self.B_params = list(model.get_B_parameter_groups())
            cls_params = []
            if hasattr(model, 'classifier'):
                cls_params = list(model.classifier.parameters())
            elif hasattr(model, 'fc'):
                cls_params = list(model.fc.parameters())
            elif hasattr(model, 'head'):
                cls_params = list(model.head.parameters())

            self.local_params = self.B_params + cls_params
            
            # Safety check for local parameters
            if len(self.local_params) == 0:
                print(f"Warning: client {client_id} has no local_params (B+classifier). Check head detection.")
                print(f"  Model has: classifier={hasattr(model, 'classifier')}, fc={hasattr(model, 'fc')}, head={hasattr(model, 'head')}")

            # Create optimizers for A-only and B+classifier
            self.dp_optimizer = torch.optim.SGD(self.A_params, **opt_kwargs)
            self.local_optimizer = torch.optim.SGD(self.local_params, **opt_kwargs)

            if OPACUS_AVAILABLE:
                self.privacy_engine = PrivacyEngine()
                # Store privacy config but don't call make_private yet (need dataloader)
                self.dp_noise_multiplier = config.get('privacy', {}).get('noise_multiplier', 1.0)
                self.dp_max_grad_norm = config.get('privacy', {}).get('max_grad_norm', 1.0)
                self.privacy_engine_attached = False  # Will attach during first train() call
                print(f"Client {client_id}: Opacus initialized, will attach during training")
            else:
                raise RuntimeError("Opacus not available")
            
            # For compatibility, set optimizer to None when using separated optimizers
            self.optimizer = None
        else:
            # Standard single optimizer for all trainable parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            print(f"Client {client_id}: Single optimizer tracking {len(trainable_params)} trainable parameters")
            
            self.optimizer = torch.optim.SGD(trainable_params, **opt_kwargs)
            
            # No DP or separated optimizers
            self.dp_optimizer = None
            self.local_optimizer = None
            self.privacy_engine = None
            self.privacy_engine_attached = False
        
        # Track local data size for weighted aggregation
        self.local_data_size = 0
        
        # privacy_mechanismをセッターで設定
        if privacy_mechanism is not None:
            self.set_privacy_mechanism(privacy_mechanism)
        
        # Initial cleanup: ensure model starts without any grad_sample artifacts
        comprehensive_grad_sample_cleanup(self.model, verbose=False)
    
    def _unwrap(self):
        """Unwrap Opacus GradSampleModule to access original model methods"""
        # Opacus 1.x wraps model in GradSampleModule which has _module attribute
        return getattr(self.model, "_module", self.model)
    
    def train(self, dataloader, training_config):
        """Train the model with A-only DP-SGD (via Opacus) and non-DP B+classifier"""
        self.model.train()
        self.local_data_size = len(dataloader.dataset)
        
        # Attach Opacus privacy engine on first training call (now we have dataloader)
        if (self.use_dp and self.aggregation_method == 'fedsa_shareA_dp' and 
            hasattr(self, 'privacy_engine') and not self.privacy_engine_attached):
            # Make the entire model private, but only A parameters will be modified by DP optimizer
            self.model, self.dp_optimizer, _ = self.privacy_engine.make_private(
                module=self.model,  # Pass the entire model to Opacus for proper hook attachment
                optimizer=self.dp_optimizer,  # A-only optimizer
                data_loader=dataloader,
                noise_multiplier=self.dp_noise_multiplier,
                max_grad_norm=self.dp_max_grad_norm,
                poisson_sampling=False
            )
            # Note: We ignore the returned dataloader and use the original one to avoid confusion
            self.privacy_engine_attached = True
            print(f"Client {self.client_id}: Opacus attached to model with A-only optimizer (batch_size={dataloader.batch_size})")
            print(f"  Note: Using original dataloader, ignoring Opacus-wrapped dataloader")

        total_loss = 0.0
        correct = 0
        total = 0
        num_epochs = training_config.get('epochs', 5)

        # AMP は DP 無効時のみ
        model_is_half = next(self.model.parameters()).dtype == torch.float16
        scaler = torch.amp.GradScaler('cuda') if (self.use_amp and torch.cuda.is_available() and not model_is_half) else None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader,
                        desc=f"Client {self.client_id} - Epoch {epoch+1}/{num_epochs}",
                        leave=False, unit="batch")

            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                if model_is_half:
                    data = data.half()

                # Mixup/CutMix の処理
                if len(target.shape) > 1:
                    target_for_loss = target
                    target_for_acc = target.argmax(dim=1)
                else:
                    target_for_loss = target
                    target_for_acc = target

                if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp' and self.privacy_engine_attached:
                    # --- DP-A + non-DP B/classifier ---
                    self.dp_optimizer.zero_grad()
                    self.local_optimizer.zero_grad()

                    # DP の場合は AMP を使わない
                    output = self.model(data)
                    if len(target.shape) > 1:
                        loss = -(target_for_loss * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, target_for_loss)

                    loss.backward()

                    # A は Opacus がノイズ注入＆クリップ＆会計
                    self.dp_optimizer.step()
                    # B + classifier は通常 SGD
                    self.local_optimizer.step()

                elif self.aggregation_method == 'fedsa_shareA_dp':
                    # 非DP の場合（2 optimizer だがノイズなし）
                    self.dp_optimizer.zero_grad()
                    self.local_optimizer.zero_grad()

                    with torch.amp.autocast('cuda', enabled=self.use_amp and scaler is not None):
                        output = self.model(data)
                    if len(target.shape) > 1:
                        loss = -(target_for_loss * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, target_for_loss)

                    loss.backward()
                    self.dp_optimizer.step()
                    self.local_optimizer.step()

                else:
                    # 標準 FedSA/FedAvg の場合（単一 optimizer）
                    self.optimizer.zero_grad()
                    with torch.amp.autocast('cuda', enabled=self.use_amp and scaler is not None):
                        output = self.model(data)
                    if len(target.shape) > 1:
                        loss = -(target_for_loss * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, target_for_loss)
                    loss.backward()
                    self.optimizer.step()

                # --- metrics ---
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_for_acc.view_as(pred)).sum().item()
                total += target_for_acc.size(0)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

            total_loss += epoch_loss
            print(f"Client {self.client_id} - Epoch {epoch+1} completed")
            
            # Epoch-level memory cleanup
            # Note: We cannot clear grad_sample during training as Opacus needs it
            # Only clear GPU cache which is safe
            if torch.cuda.is_available():
                if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp':
                    # More aggressive GPU cleanup for DP mode
                    torch.cuda.empty_cache()
                elif epoch % 2 == 1:
                    # Lighter cleanup for non-DP modes
                    torch.cuda.empty_cache()

        avg_loss = total_loss / (num_epochs * len(dataloader))
        accuracy = 100. * correct / total

        # Comprehensive cleanup after training (memory efficiency)
        # This is safe to do after all epochs are complete
        if self.use_dp and self.aggregation_method == 'fedsa_shareA_dp':
            # For DP mode, use Opacus's zero_grad to properly clear gradients
            self.dp_optimizer.zero_grad(set_to_none=True)
            
            # Then do comprehensive cleanup
            comprehensive_grad_sample_cleanup(self.model, verbose=False)
        else:
            # Standard cleanup for non-DP
            safe_clear_grad_sample(self.model)
        
        # --- FedSA 用: A だけ返す ---
        update = {
            'A_params': self._unwrap().get_A_parameters(),
            'num_samples': self.local_data_size,
            'local_data_size': self.local_data_size,
            'loss': avg_loss,
            'accuracy': accuracy,
            'upload_type': 'A_matrices_only'
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
        """Update model with global A parameters (FedSA-LoRA)"""
        # Only update A parameters, keep B local
        if 'A_params' in global_params:
            self._unwrap().set_A_parameters(global_params['A_params'])
            print(f"Client {self.client_id}: Updated A matrices from server")
            
            # Note: self.model.set_A_parameters() uses copy_() which preserves parameter identity
            # This means optimizer state (momentum, etc.) is maintained across rounds
            # No need to reset optimizer state here
            
            # NOTE: We don't need to reset optimizer state here because:
            # 1. The copy_() method in set_A_parameters preserves parameter identity
            # 2. This allows momentum to be maintained across rounds
            # 3. Resetting here would cause KeyError in the next training round
    
    def reset_A_optimizer_state(self):
        """Reset optimizer state for A parameters after server update"""
        # No-op: State preservation is handled by parameter identity preservation in set_A_parameters
        # The copy_() method in set_A_parameters preserves parameter identity,
        # allowing momentum and other optimizer states to continue across rounds
        pass
    
    def get_model_size(self):
        """Get model size information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.aggregation_method == 'fedsa_shareA_dp':
            A_params = sum(p.numel() for p in self._unwrap().get_A_parameter_groups())
            B_params = sum(p.numel() for p in self._unwrap().get_B_parameter_groups())
            communication_params = A_params  # Only A is communicated
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
    # Initialize Slack notifier
    slack_notifier = None
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if webhook_url:
        slack_notifier = SlackNotifier(webhook_url)
        print("Slack notifications enabled")
    else:
        print("Slack notifications disabled (set SLACK_WEBHOOK_URL environment variable to enable)")

    parser = argparse.ArgumentParser(description='FedSA-LoRA ResNet (A matrices only)')
    parser.add_argument('--config', type=str, default='configs/cifar100_mobilenet.yaml',
                       help='Path to configuration file')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Override number of rounds')
    parser.add_argument('--clients', type=int, default=None,
                       help='Override number of clients')
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 
                       default=None, help='Override ResNet model variant')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Creating default configuration for MobileNetV2 on CIFAR-100 with transfer learning...")
        
        # Create default config if not exists
        default_config = {
            'seed': 42,
            'use_gpu': True,
            'data': {
                'dataset_name': 'cifar100',
                'data_dir': './data',
                'num_clients': 10,
                'data_split': 'non_iid',
                'alpha': 0.5,
                'batch_size': 64,
                'num_workers': 4,
                'model_type': 'imagenet',  # Use ImageNet model for transfer learning
                'imagenet_style': True,  # Enable ImageNet-style preprocessing
                'input_size': 224,  # ImageNet input size
                'verbose': False,
                'augmentations': {
                    'horizontal_flip': {'enabled': True, 'prob': 0.5},
                    'random_rotation': {'enabled': True, 'degrees': 10},
                    'color_jitter': {
                        'enabled': False,
                        'brightness': 0.2,
                        'contrast': 0.2,
                        'saturation': 0.2,
                        'hue': 0.1
                    },
                    'random_crop': {'enabled': False, 'padding': 4},
                    'random_resized_crop': {'enabled': True, 'scale_min': 0.5},  # For ImageNet-style
                    'random_erasing': {'enabled': False, 'prob': 0.5},
                    'mixup': {'enabled': False, 'alpha': 0.2, 'prob': 0.5},
                    'cutmix': {'enabled': False, 'alpha': 1.0, 'prob': 0.5}
                }
            },
            'model': {
                'model_name': 'mobilenet_v2',  # Changed to MobileNetV2 for efficiency
                'num_classes': 100,
                'pretrained': True,  # Use ImageNet pretrained weights
                'freeze_backbone': True,  # Freeze backbone for transfer learning
                'lora': {  # LoRA configuration structure
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
                'num_rounds': 100,
                'num_clients': 10,
                'client_fraction': 1.0,
                'aggregation_method': 'fedsa',  # Use FedSA for LoRA
                'checkpoint_freq': 20
            },
            'privacy': {
                'enable_privacy': False,
                'epsilon': 10.0,
                'delta': 1e-5,
                'max_grad_norm': 0.5
            },
            'evaluation': {
                'eval_freq': 5
            },
            'experiment': {
                'name': 'MobileNetV2_CIFAR100_NonIID_Transfer',
                'output_dir': 'experiments/quickstart_resnet'
            },
            'reproducibility': {
                'deterministic': False
            }
        }
        
        # Save default config
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        config_path = configs_dir / "cifar100_mobilenet.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        print(f"Default configuration saved to: {config_path}")
        config = default_config
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override configuration with command line arguments
    if args.rounds:
        config['federated']['num_rounds'] = args.rounds
    if args.clients:
        config['federated']['num_clients'] = args.clients
        config['data']['num_clients'] = args.clients
    if args.model:
        config['model']['model_name'] = args.model
    
    # Set device
    device = torch.device('cuda' if config.get('use_gpu', False) and torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("FedSA-LoRA ResNet Federated Learning (A matrices only)")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Dataset: {config['data']['dataset_name'].upper()}")
    print(f"Clients: {config['federated']['num_clients']}")
    print(f"Rounds: {config['federated']['num_rounds']}")
    privacy_enabled = config.get('privacy', {}).get('enable_privacy', False)
    opacus_flag = config.get('privacy', {}).get('use_opacus_accounting', True)
    print(f"Privacy: {'Enabled' if privacy_enabled else 'Disabled'}")
    if privacy_enabled:
        agg_method = config['federated'].get('aggregation_method', 'fedsa')
        if agg_method == 'fedsa_shareA_dp':
            print("Privacy Method: Opacus DP-SGD (A-only, RDP accountant)")
        else:
            print("Privacy Method: Opacus DP-SGD (RDP accountant)")
    print("=" * 80)
    
    # Set seed comprehensively for reproducibility
    if 'seed' in config:
        seed = config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Optional: for deterministic behavior (may reduce performance)
        if config.get('reproducibility', {}).get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    
    # Check Opacus availability if privacy is enabled
    if config.get('privacy', {}).get('enable_privacy', False):
        if not OPACUS_AVAILABLE:
            raise RuntimeError(
                "Privacy is enabled but Opacus is not installed. "
                "Install with: pip install opacus\n"
                "Or disable privacy by setting privacy.enable_privacy: false"
            )
        print("✅ Opacus is available for Differential Privacy")
    
    # Prepare data with appropriate transforms
    print("\nPreparing federated data...")
    
    # Configure data transforms based on model type
    use_imagenet_style = config['data'].get('imagenet_style', False)
    if use_imagenet_style:
        print("Using ImageNet-style transforms (224x224)")
        config['data']['model_type'] = 'imagenet'
        config['data']['use_cifar_resnet'] = False
    else:
        print("Using CIFAR-optimized transforms (32x32)")
        config['data']['model_type'] = 'resnet'
        config['data']['use_cifar_resnet'] = True
    
    trainset, testset, client_train_indices, client_test_indices = prepare_federated_data(config['data'])
    
    # Prepare Mixup/CutMix if enabled
    collate_fn = None
    augmentation_config = config['data'].get('augmentations', {})
    mixup_config = augmentation_config.get('mixup', {})
    cutmix_config = augmentation_config.get('cutmix', {})
    
    # Determine number of classes
    num_classes = config['model'].get('num_classes', 100)
    
    if mixup_config.get('enabled', False) or cutmix_config.get('enabled', False):
        print("Enabling Mixup/CutMix augmentation...")
        collate_fn = MixupCutmixCollate(
            mixup_alpha=mixup_config.get('alpha', 0.2) if mixup_config.get('enabled', False) else 0,
            cutmix_alpha=cutmix_config.get('alpha', 1.0) if cutmix_config.get('enabled', False) else 0,
            mixup_prob=mixup_config.get('prob', 0.5) if mixup_config.get('enabled', False) else 0,
            cutmix_prob=cutmix_config.get('prob', 0.5) if cutmix_config.get('enabled', False) else 0,
            num_classes=num_classes
        )
        collate_fn.set_training(True)  # Enable for training
    
    # Create test dataloader
    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True
    )
    
    # Create initial global model
    use_imagenet_style = config['data'].get('imagenet_style', False)
    
    if use_imagenet_style:
        # Use ImageNet-style backbone with LoRA
        print(f"Creating ImageNet-style {config['model']['model_name']} model...")
        initial_model = make_model_with_lora(config)
        initial_model = initial_model.to(device)  # Move to GPU
        print_model_summary(config, initial_model)
        
        # Debug: Check if pretrained weights are loaded
        print(f"🔍 Transfer Learning Check:")
        print(f"  Model name: {config['model']['model_name']}")
        print(f"  Pretrained: {config['model'].get('pretrained', True)}")
        print(f"  Freeze backbone: {config['model'].get('freeze_backbone', True)}")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in initial_model.parameters())
        trainable_params = sum(p.numel() for p in initial_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        
        if trainable_params / total_params > 0.5:
            print(f"  ⚠️  WARNING: Too many trainable params! Transfer learning may not be working properly.")
    else:
        # Use CIFAR-optimized ResNet with LoRA (legacy)
        print(f"Creating CIFAR-optimized {config['model']['model_name']} model with LoRA (32x32 input)...")
        model_config = config['model'].copy()
        model_config['seed'] = config.get('seed', 42)
        initial_model = create_cifar_resnet_lora(model_config)
    
    # Initialize server
    print("Initializing federated server...")
    server = FedSAFTLServer(device)
    # Set initial global A parameters for synchronization
    server.global_A_params = initial_model.get_A_parameters()
    print(f"✅ Server initialized with global A matrices from seed {config.get('seed', 42)}")
    
    # Create temp client for statistics
    temp_client = ResNetFedSAFTLClient(0, initial_model, device, config)
    model_stats = temp_client.get_model_size()
    
    print("\nModel Statistics:")
    print(f"  Architecture: {config['model']['model_name']}")
    print(f"  Total parameters: {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    # Handle both old and new config formats
    lora_rank = config['model'].get('lora', {}).get('r', config['model'].get('lora_r', 0))
    if lora_rank > 0:
        print(f"  LoRA rank: {lora_rank}")
    print(f"  Communication overhead: {model_stats['communication_params']:,} parameters/round")
    print(f"  Compression ratio: {model_stats['compression_ratio']:.2f}x")
    
    # Create clients
    print(f"\nCreating {config['federated']['num_clients']} MobileNetV2 federated clients with transfer learning...")
    clients = []
    
    
    # Create results directory with date and ResNet identifier
    current_date = datetime.now().strftime('%m%d')  # MMDD format
    base_experiment_dir = Path(config.get('experiment', {}).get('output_dir', 'experiments/quickstart_resnet'))
    # Create method-specific subdirectory (e.g., quickstart_resnet_iid/fedsa or /fedsa_shareA_dp)
    agg_method = config.get('federated', {}).get('aggregation_method', 'unknown')
    method_subdir = agg_method.lower()
    method_dir = base_experiment_dir / method_subdir
    experiment_dir = method_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure directory exists
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {experiment_dir} (aggregation_method={agg_method})")
    
    # Generate file names with date and ResNet identifier
    date_resnet_suffix = f"{current_date}ResNet"
    
    # Initialize results tracking
    results = {
        'config': config,
        'start_time': datetime.now().isoformat(),
        'experiment_dir': str(experiment_dir),
        'date_suffix': date_resnet_suffix,
        'rounds': [],
        'summary': {}
    }
    
    # Send training start notification
    if slack_notifier:
        config_for_notification = {
            'experiment': {'name': f'ResNet QuickStart ({config["model"]["model_name"]})'},
            'federated': config['federated'],
            'privacy': config.get('privacy', {})
        }
        slack_notifier.send_training_start(config_for_notification)
    
    # DPが有効かどうかをループの前に一度だけチェック
    privacy_enabled = config.get('privacy', {}).get('enable_privacy', False)
    
    for client_id in range(config['federated']['num_clients']):
        privacy_mechanism = None
        # ループ内でクライアントごとに新しいインスタンスを生成
        if privacy_enabled:
            privacy_mechanism = DifferentialPrivacy(
                epsilon=config['privacy'].get('epsilon', 10.0),
                delta=config['privacy'].get('delta', 1e-5),
                max_grad_norm=config['privacy'].get('max_grad_norm', 0.5),
                total_rounds=config['federated'].get('num_rounds', 100)
            )
        
        # Create client model (same architecture as initial model)
        if use_imagenet_style:
            client_model = make_model_with_lora(config)
            client_model = client_model.to(device)  # Move to GPU
        else:
            model_config = config['model'].copy()
            model_config['seed'] = config.get('seed', 42)
            client_model = create_cifar_resnet_lora(model_config)
        
        # Synchronize with server's initial parameters
        if hasattr(server, 'global_A_params'):
            if hasattr(client_model, 'set_A_parameters'):
                client_model.set_A_parameters(server.global_A_params)
            else:
                # For ImageNet models, load full state dict (can be optimized later)
                client_model.load_state_dict(initial_model.state_dict())
        
        client = ResNetFedSAFTLClient(
            client_id, 
            client_model, 
            device,
            config,  # Pass config for DP and aggregation settings
            privacy_mechanism  # 各クライアントに固有のオブジェクトを渡す
        )
        clients.append(client)
    
    print(f"✅ All {len(clients)} clients initialized with synchronized A matrices (B matrices remain local)")
    print("Starting MobileNetV2 federated training with transfer learning...")
    print("=" * 80)
    
    # Training loop
    best_accuracy = 0
    best_round = 0
    start_time = time.time()
    
    # Create main progress bar for rounds
    round_pbar = tqdm(range(config['federated']['num_rounds']), 
                     desc="Federated Rounds",
                     unit="round")
    
    for round_idx in round_pbar:
        print(f"\n[Round {round_idx + 1}/{config['federated']['num_rounds']}]")
        round_start_time = time.time()
        
        # Select clients based on client_fraction
        client_fraction = config['federated'].get('client_fraction', 1.0)
        num_clients = config['federated']['num_clients']
        num_selected = max(1, int(np.ceil(client_fraction * num_clients)))
        
        if client_fraction >= 1.0:
            # Full participation
            selected_clients = list(range(num_clients))
        else:
            # Random sampling
            selected_clients = sorted(random.sample(range(num_clients), num_selected))
        
        print(f"Selected clients: {selected_clients}")
        
        # Client updates
        client_updates = []
        train_accuracies = []
        
        # Create progress bar for client training
        client_pbar = tqdm(selected_clients, 
                          desc="Training clients",
                          leave=False,
                          unit="client")
        
        for client_id in client_pbar:
            client_pbar.set_description(f"Training client {client_id}")
            # Get client training data with optional Mixup/CutMix
            client_dataloader = get_client_dataloader(
                trainset,
                client_train_indices[client_id],
                config['data']['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,  # Use Mixup/CutMix if enabled
                num_workers=config['data'].get('num_workers', 0)
            )
            
            # Update with global A parameters (FedSA-LoRA)
            if hasattr(server, 'global_A_params') and server.global_A_params:
                clients[client_id].update_model({'A_params': server.global_A_params})
            
            # Local training
            client_result = clients[client_id].train(client_dataloader, config['training'])
            client_updates.append(client_result)
            train_accuracies.append(client_result['accuracy'])
            
            print(f"  Client {client_id}: Loss={client_result['loss']:.4f}, "
                  f"Accuracy={client_result['accuracy']:.2f}%")
            
            # Per-client cleanup to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Log GPU memory if needed for debugging
                if (round_idx + 1) % 10 == 0:  # Every 10 rounds
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    cached = torch.cuda.memory_reserved() / 1024**3  # GB
                    print(f"    GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # Evaluation
        is_eval_round = (round_idx + 1) % config.get('evaluation', {}).get('eval_freq', 5) == 0
        
        if is_eval_round:
            print("\nEvaluating models...")
            
            # 1. Personalized Accuracy: Each client evaluates on their LOCAL test data
            print("  Personalized Accuracy (local test data):")
            personalized_results = []
            personalized_accuracies = []
            for client_id in selected_clients:
                # Get client's LOCAL test data (no augmentation for testing)
                client_test_dataloader = get_client_dataloader(
                    testset,
                    client_test_indices[client_id],
                    config['data']['batch_size'],
                    shuffle=False,
                    collate_fn=None,  # No Mixup/CutMix for testing
                    num_workers=config['data'].get('num_workers', 0)
                )
                # Evaluate personalized model (A * B_i) on local test data
                test_result = clients[client_id].evaluate(client_test_dataloader)
                personalized_results.append(test_result)
                personalized_accuracies.append(test_result['accuracy'])
                print(f"    Client {client_id}: {test_result['accuracy']:.2f}%")
            
            avg_personalized_acc = sum(personalized_accuracies) / len(personalized_accuracies)
            print(f"  Average Personalized Accuracy: {avg_personalized_acc:.2f}%")
            
            # Use personalized results for server aggregation
            client_test_results = personalized_results
            test_accuracies = personalized_accuracies
        else:
            test_accuracies = None  # No evaluation this round
            personalized_accuracies = None
            avg_personalized_acc = None
            # Create proper test results with both accuracy and loss keys
            client_test_results = [{'accuracy': 0, 'loss': 0} for _ in selected_clients]
        
        # FedSA-LoRA Server aggregation (A matrices only)
        aggregation_method = config['federated'].get('aggregation_method', 'fedsa')
        
        if aggregation_method not in ['fedsa_shareA_dp', 'fedsa']:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}. This script only supports 'fedsa' and 'fedsa_shareA_dp'. Use quickstart_resnet_fedavg.py for standard FedAvg.")
        
        # FedSA: aggregate only A matrices
        client_A_params = [update['A_params'] for update in client_updates]
        client_sample_counts = [update['local_data_size'] for update in client_updates]
        
        # Aggregate A matrices using weighted average
        aggregated_A = WeightedFedAvg.aggregate_A_matrices(client_A_params, client_sample_counts)
        
        # Store global A parameters
        if not hasattr(server, 'global_A_params'):
            server.global_A_params = {}
        server.global_A_params = aggregated_A
        
        # Log aggregation info
        WeightedFedAvg.log_aggregation_info(client_sample_counts, len(aggregated_A))
        
        # Round-level cleanup to prevent memory leaks
        if round_idx % 5 == 0:  # Every 5 rounds
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Additional cleanup for all clients
            for client in clients:
                comprehensive_grad_sample_cleanup(client.model, verbose=False)
            
            if (round_idx + 1) % 10 == 0:  # Every 10 rounds, show memory info
                print(f"  🧹 Memory cleanup completed (Round {round_idx + 1})")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    cached = torch.cuda.memory_reserved() / 1024**3
                    print(f"  GPU Memory after cleanup: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # Calculate communication cost (only A matrices, dtype-aware)
        def get_bytes_per_param(tensor):
            """Get bytes per parameter based on dtype"""
            if tensor.dtype in (torch.float16, torch.bfloat16):
                return 2
            else:  # float32, int32, etc.
                return 4
        
        total_bytes = sum(p.numel() * get_bytes_per_param(p) for p in aggregated_A.values())
        communication_cost_mb = total_bytes / (1024 * 1024)
        
        # Also track parameter count for statistics
        total_A_params = sum(p.numel() for p in aggregated_A.values())
        
        # Verify only A parameters were uploaded
        for i, update in enumerate(client_updates):
            if 'upload_type' in update and update['upload_type'] != 'A_matrices_only':
                print(f"WARNING: Client {i} uploaded unexpected parameters!")
        
        # Enhanced privacy information logging
        current_epsilon = None  # Initialize for CSV output
        if config.get('privacy', {}).get('enable_privacy', False):
            privacy_analyses = [update.get('privacy_analysis', {}) for update in client_updates if 'privacy_analysis' in update]
            if privacy_analyses:
                # Average privacy metrics across clients (using unified 'epsilon' key)
                epsilons = [analysis.get('epsilon', 0) for analysis in privacy_analyses if 'epsilon' in analysis]
                
                if epsilons:
                    avg_eps = sum(epsilons) / len(epsilons)
                    max_eps = max(epsilons)  # Report worst-case for safety
                    current_epsilon = max_eps  # Use max epsilon for conservative tracking
                    
                    print(f"\n  === Privacy Analysis (Opacus DP-SGD A-only) ===")
                    print(f"  Current ε (avg): {avg_eps:.4f}, ε (max): {max_eps:.4f}")
                    print(f"  Privacy Budget: ε≤{config['privacy'].get('epsilon', 8.0)}")
                    print(f"  δ: {config['privacy'].get('delta', 1e-5)}")
                    
                    # Show key privacy parameters from first client
                    sample_analysis = privacy_analyses[0]
                    print(f"  Noise multiplier (σ): {sample_analysis.get('noise_multiplier', 'N/A')}")
                    print(f"  Max gradient norm (C): {sample_analysis.get('max_grad_norm', 'N/A')}")
                    print(f"  Batch size: {sample_analysis.get('batch_size', 'N/A')}")
                    print(f"  Sample rate: {sample_analysis.get('sample_rate', 'N/A'):.4f}" if sample_analysis.get('sample_rate') else "  Sample rate: N/A")
                    print(f"  Method: {sample_analysis.get('method', 'Opacus DP-SGD (A-only)')}")
                    print(f"  Protection: LoRA-A matrices only (shared parameters)")
                    print(f"  Non-DP: LoRA-B + classifier (local personalized parameters)")
                    print(f"  Note: Sample rate shown is approx (batch_size/dataset_size);")
                    print(f"        exact value is managed by Opacus accountant")
                else:
                    print(f"\n  === Privacy Analysis ===")
                    print(f"  Privacy tracking unavailable - check Opacus installation")
                print(f"  =============================================")
        
        round_stats = {
            'communication_cost_mb': communication_cost_mb,
            'aggregated_params': total_A_params,
            'aggregation_method': aggregation_method
        }
        
        # Print summary with timing
        round_time = time.time() - round_start_time
        total_time = time.time() - start_time
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)
        
        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Avg Training Accuracy: {avg_train_acc:.2f}%")
        
        # Track best accuracy only during evaluation rounds
        is_new_best = False
        if is_eval_round and test_accuracies is not None:
            print(f"  Avg Personalized Test Accuracy: {avg_personalized_acc:.2f}%")
            # Track best PERSONALIZED accuracy as the main metric
            if avg_personalized_acc > best_accuracy:
                best_accuracy = avg_personalized_acc
                best_round = round_idx + 1
                is_new_best = True
                print(f"  ** New best personalized accuracy! **")
        
        print(f"  Communication Cost (per-round): {round_stats.get('communication_cost_mb', 0):.2f} MB")
        print(f"  Round time: {round_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        
        # Estimate remaining time
        rounds_completed = round_idx + 1
        if rounds_completed > 0:
            avg_round_time = total_time / rounds_completed
            remaining_rounds = config['federated']['num_rounds'] - rounds_completed
            estimated_remaining = avg_round_time * remaining_rounds
            print(f"  Estimated remaining time: {estimated_remaining:.0f}s ({estimated_remaining/60:.1f} min)")
        
        # Update main progress bar
        round_pbar.set_postfix({
            'train_acc': f'{avg_train_acc:.2f}%',
            'test_acc': f'{avg_personalized_acc:.2f}%' if avg_personalized_acc else 'N/A',
            'best': f'{best_accuracy:.2f}%',
            'time': f'{round_time:.1f}s'
        })
        
        # Save round results
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
        
        # Add epsilon if using differential privacy
        if current_epsilon is not None:
            round_result['epsilon'] = current_epsilon
        
        results['rounds'].append(round_result)
        
        # Save results to file after each round (with date+ResNet suffix)
        results_file = experiment_dir / f'training_results_{date_resnet_suffix}.json'
        
        # Ensure parent directory exists
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save results file: {e}")
            print(f"Attempted to save to: {results_file}")
        
        # Send Slack notification every 10 rounds
        if slack_notifier and (round_idx + 1) % 10 == 0:
            config_for_notification = {
                'experiment': {'name': f'ResNet QuickStart ({config["model"]["model_name"]})'},
                'federated': config['federated']
            }
            
            # Prepare round stats for notification
            round_stats_for_notification = {
                'train_accuracy': avg_train_acc,
                'test_accuracy': avg_personalized_acc if avg_personalized_acc is not None else None,
                'per_round_communication_mb': round_stats.get('communication_cost_mb', 0)
            }
            
            # Prepare server summary for notification
            server_summary_for_notification = {
                'best_test_accuracy': best_accuracy
            }
            
            slack_notifier.send_progress_update(
                config_for_notification,
                round_idx,
                round_stats_for_notification,
                server_summary_for_notification
            )
        
        # Checkpoint
        if (round_idx + 1) % config['federated'].get('checkpoint_freq', 20) == 0:
            print(f"  Checkpoint saved at round {round_idx + 1}")
            # Save checkpoint results to separate file (with date+ResNet suffix)
            checkpoint_file = experiment_dir / f'checkpoint_round_{round_idx + 1}_{date_resnet_suffix}.json'
            checkpoint_data = {
                'round': round_idx + 1,
                'best_accuracy': best_accuracy,
                'best_round': best_round,
                'current_accuracy': avg_personalized_acc,
                'recent_rounds': results['rounds'][-10:]  # Last 10 rounds
            }
            
            # Ensure parent directory exists and save with error handling
            try:
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                print(f"  Checkpoint saved to: {checkpoint_file}")
            except Exception as e:
                print(f"  Warning: Failed to save checkpoint: {e}")
    
    # Final results
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Complete results summary
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
        'dataset': config['data']['dataset_name']
    }
    
    # Save final results (with date+ResNet suffix)
    final_results_file = experiment_dir / f'final_results_{date_resnet_suffix}.json'
    
    try:
        final_results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {final_results_file}")
    except Exception as e:
        print(f"Warning: Failed to save final results: {e}")
    
    # Create summary CSV for easy plotting (with date+ResNet suffix)
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
            # Add epsilon column if available
            if 'epsilon' in round_data:
                row_data['epsilon'] = round_data['epsilon']
            df_data.append(row_data)
        
        if df_data:  # Only create CSV if we have data
            df = pd.DataFrame(df_data)
            csv_file = experiment_dir / f'results_summary_{date_resnet_suffix}.csv'
            df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved to: {csv_file}")
        else:
            print("Warning: No round data available for CSV export")
            csv_file = None
            
    except ImportError:
        print("Warning: pandas not available, skipping CSV export")
        csv_file = None
    except Exception as e:
        print(f"Warning: Failed to create CSV summary: {e}")
        csv_file = None
    
    print("\n" + "=" * 80)
    print("ResNet Federated Learning Complete!")
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
    print(f"  - Training results: final_results_{date_resnet_suffix}.json")
    if csv_file:
        print(f"  - Summary CSV: results_summary_{date_resnet_suffix}.csv")
    print(f"  - File naming pattern: *_{date_resnet_suffix}.*")
    print("=" * 80)
    
    # Send completion notification
    if slack_notifier:
        config_for_notification = {
            'experiment': {
                'name': f'ResNet QuickStart ({config["model"]["model_name"]})',
                'output_dir': str(experiment_dir)
            },
            'federated': config['federated'],
            'privacy': config.get('privacy', {})
        }
        
        total_comm_final = sum(r['communication_cost_mb'] for r in results['rounds'])
        avg_per_round_final = total_comm_final / len(results['rounds']) if results['rounds'] else 0
        
        summary_for_notification = {
            'final_avg_accuracy': avg_personalized_acc if 'avg_personalized_acc' in locals() else 0,
            'final_std_accuracy': 0,  # Can calculate from individual client results if needed
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
    
    # Evaluation
    if best_accuracy > 20:  # Reasonable threshold for CIFAR-100
        print("🎉 MobileNetV2 federated transfer learning successful!")
    elif best_accuracy > 10:
        print("⚠️ MobileNetV2 transfer learning in progress - consider more rounds")
    else:
        print("❌ MobileNetV2 transfer learning may need hyperparameter tuning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error notification if Slack is configured
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        if webhook_url:
            try:
                slack_notifier = SlackNotifier(webhook_url)
                config_for_notification = {
                    'experiment': {'name': 'ResNet QuickStart (ERROR)'}
                }
                slack_notifier.send_error_notification(
                    config_for_notification,
                    str(e) + "\n" + traceback.format_exc()[:500]
                )
            except Exception:
                pass  # Don't fail on notification failure