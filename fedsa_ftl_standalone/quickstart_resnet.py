#!/usr/bin/env python3
"""
Quick start script for ResNet federated learning on CIFAR-100
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
from fedsa_ftl_model_resnet import create_model_resnet
from fedsa_ftl_client import FedSAFTLClient
from fedsa_ftl_server import FedSAFTLServer
from data_utils import prepare_federated_data, get_client_dataloader, MixupCutmixCollate
from privacy_utils import DifferentialPrivacy
from notification_utils import SlackNotifier
from dp_utils import create_dp_optimizer, WeightedFedAvg
import torch.nn.functional as F


class ResNetFedSAFTLClient(FedSAFTLClient):
    """Extended client for ResNet models with DP-LoRA support"""
    
    def __init__(self, client_id, model, device, config, privacy_mechanism=None):
        super().__init__(client_id, model, device)
        self.config = config
        self.use_dp = config.get('privacy', {}).get('enable_privacy', False)
        self.aggregation_method = config.get('federated', {}).get('aggregation_method', 'fedavg')
        
        # Initialize optimizer based on aggregation method and privacy settings
        if self.aggregation_method == 'fedsa_shareA_dp':
            if self.use_dp:
                # DP optimizer for A matrices only - will update dataset_size in train()
                self.dp_optimizer = create_dp_optimizer(
                    model, config, 
                    batch_size=config.get('data', {}).get('batch_size', 64),
                    dataset_size=1000  # Placeholder, updated in train()
                )
                self.optimizer = None  # Not used when DP is enabled
                print(f"Client {client_id}: Initialized DP optimizer for A matrices only")
            else:
                # Regular optimizer for FedSA without DP
                training_config = config.get('training', {})
                self.optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=training_config.get('lr', 0.001),
                    momentum=training_config.get('momentum', 0.9),
                    weight_decay=training_config.get('weight_decay', 0.0001)
                )
                self.dp_optimizer = None
        else:
            # Standard federated learning optimizer
            training_config = config.get('training', {})
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=training_config.get('lr', 0.001),
                momentum=training_config.get('momentum', 0.9),
                weight_decay=training_config.get('weight_decay', 0.0001)
            )
            self.dp_optimizer = None
        
        # Track local data size for weighted aggregation
        self.local_data_size = 0
        
        # privacy_mechanismをセッターで設定
        if privacy_mechanism is not None:
            self.set_privacy_mechanism(privacy_mechanism)
    
    def train(self, dataloader, training_config):
        """Train the model with DP-LoRA support"""
        self.model.train()
        self.local_data_size = len(dataloader.dataset)
        
        # Update DP optimizer with actual dataset size for privacy accounting
        if hasattr(self, 'dp_optimizer') and self.dp_optimizer is not None:
            self.dp_optimizer.update_dataset_size(self.local_data_size)
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_epochs = training_config.get('epochs', 5)
        microbatch_size = training_config.get('microbatch_size', 8)  # For DP
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Handle Mixup/CutMix targets
                if len(target.shape) > 1:  # One-hot encoded (from Mixup/CutMix)
                    target_for_loss = target
                    target_for_acc = target.argmax(dim=1)
                else:
                    target_for_loss = target
                    target_for_acc = target
                
                output = self.model(data)
                
                # Calculate loss with proper reduction for DP
                if self.aggregation_method == 'fedsa_shareA_dp' and self.dp_optimizer is not None:
                    # DP requires per-sample losses for proper clipping
                    if len(target.shape) > 1:  # Mixup/CutMix case
                        # Convert to per-sample losses
                        loss_vec = -(target_for_loss * F.log_softmax(output, dim=1)).sum(dim=1)
                    else:
                        loss_vec = F.cross_entropy(output, target_for_loss, reduction='none')
                    
                    # Efficient DP processing: single backward for both A and B
                    # A gets DP treatment, B gets regular gradients from the same loss
                    self.dp_optimizer.dp_backward_on_loss_efficient(
                        loss_vec, microbatch_size, 
                        also_compute_B_grads=True
                    )
                    
                    # Calculate mean loss for logging
                    loss = loss_vec.mean()
                    
                    # Step both optimizers (A has DP, B has regular gradients)
                    self.dp_optimizer.A_optimizer.step()
                    self.dp_optimizer.B_optimizer.step()
                    
                else:
                    # Standard training
                    if len(target.shape) > 1:  # Mixup/CutMix loss
                        loss = -(target_for_loss * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, target_for_loss)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_for_acc.view_as(pred)).sum().item()
                total += target_for_acc.size(0)
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (num_epochs * len(dataloader))
        accuracy = 100. * correct / total
        
        # Prepare update based on aggregation method
        if self.aggregation_method == 'fedsa_shareA_dp':
            # CRITICAL: Only return A parameters for FedSA - B matrices stay local!
            
            # Step 1: Get A parameter whitelist (safe keys only)
            A_whitelist = set(self.model.get_A_parameters().keys())
            all_A_params = self.model.get_A_parameters()
            
            # Step 2: Filter using whitelist (defense against future code changes)
            safe_A_params = {k: v for k, v in all_A_params.items() if k in A_whitelist}
            
            # Step 3: Double safety check: ensure no B parameters are included
            B_param_names = set(self.model.get_B_parameters().keys())
            A_param_names = set(safe_A_params.keys())
            leaked_B_params = A_param_names.intersection(B_param_names)
            
            if leaked_B_params:
                raise ValueError(f"SECURITY ALERT: B parameters leaked in A upload: {leaked_B_params}")
            
            # Step 4: Verify all A parameters are accounted for
            if len(safe_A_params) != len(A_whitelist):
                missing = A_whitelist - A_param_names
                raise ValueError(f"INTEGRITY ERROR: Missing A parameters: {missing}")
            
            update = {
                'A_params': safe_A_params,  # Use filtered safe parameters
                'local_data_size': self.local_data_size,
                'loss': avg_loss,
                'accuracy': accuracy,
                'upload_type': 'A_matrices_only',  # For verification
                'param_count': len(safe_A_params),  # For verification
                'param_names': list(safe_A_params.keys())  # For debugging
            }
            
            if self.dp_optimizer is not None:
                # Comprehensive privacy analysis for A matrices only
                privacy_analysis = self.dp_optimizer.get_privacy_analysis()
                update['privacy_spent'] = privacy_analysis.get('custom_epsilon', 0)
                update['privacy_analysis'] = privacy_analysis
                update['privacy_note'] = 'DP applied to A matrices only'
                
                # Log privacy info for this client
                custom_eps = privacy_analysis.get('custom_epsilon', 0)
                opacus_eps = privacy_analysis.get('opacus_epsilon')
                if opacus_eps != 'Not available':
                    print(f"Client {self.client_id}: Privacy ε={custom_eps:.4f} (custom), ε={opacus_eps:.4f} (Opacus)")
                else:
                    print(f"Client {self.client_id}: Privacy ε={custom_eps:.4f} (custom only, install opacus for accurate)")
            
            print(f"Client {self.client_id}: Uploading {len(safe_A_params)} A matrices only (B kept local)")
            print(f"Client {self.client_id}: A param names: {list(safe_A_params.keys())}")
        else:
            # Return all parameters for standard federated learning
            update = {
                'model_state': self.model.state_dict(),
                'local_data_size': self.local_data_size,
                'loss': avg_loss,
                'accuracy': accuracy
            }
        
        return update
    
    def update_model(self, global_params):
        """Update model with global parameters"""
        if self.aggregation_method == 'fedsa_shareA_dp':
            # Only update A parameters, keep B local
            if 'A_params' in global_params:
                self.model.set_A_parameters(global_params['A_params'])
                print(f"Client {self.client_id}: Updated A matrices from server")
                
                # CRITICAL: Reset A optimizer state after parameter update
                if hasattr(self, 'dp_optimizer') and self.dp_optimizer is not None:
                    self.reset_A_optimizer_state()
                    print(f"Client {self.client_id}: Reset A optimizer state")
        else:
            # Standard model update
            if 'model_state' in global_params:
                self.model.load_state_dict(global_params['model_state'])
    
    def reset_A_optimizer_state(self):
        """Reset optimizer state for A parameters after server update"""
        if hasattr(self, 'dp_optimizer') and self.dp_optimizer is not None:
            # Clear momentum and other state for A parameters
            for p in self.model.get_A_parameter_groups():
                if p in self.dp_optimizer.A_optimizer.state:
                    self.dp_optimizer.A_optimizer.state[p].clear()
            print(f"Client {self.client_id}: Cleared A optimizer momentum/state")
    
    def get_model_size(self):
        """Get model size information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.aggregation_method == 'fedsa_shareA_dp':
            A_params = sum(p.numel() for p in self.model.get_A_parameter_groups())
            B_params = sum(p.numel() for p in self.model.get_B_parameter_groups())
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

    parser = argparse.ArgumentParser(description='FedSA-FTL ResNet Quick Start')
    parser.add_argument('--config', type=str, default='configs/cifar100_resnet50.yaml',
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
        print("Creating default configuration for ResNet50 on CIFAR-100...")
        
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
                'model_type': 'resnet',
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
                    'random_erasing': {'enabled': False, 'prob': 0.5},
                    'mixup': {'enabled': False, 'alpha': 0.2, 'prob': 0.5},
                    'cutmix': {'enabled': False, 'alpha': 1.0, 'prob': 0.5}
                }
            },
            'model': {
                'model_name': 'resnet50',
                'num_classes': 100,
                'pretrained': True,
                'lora_r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.1
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
                'aggregation_method': 'fedavg',
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
                'name': 'ResNet50_CIFAR100_NonIID',
                'output_dir': 'experiments/quickstart_resnet'
            },
            'reproducibility': {
                'deterministic': False
            }
        }
        
        # Save default config
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        config_path = configs_dir / "cifar100_resnet50.yaml"
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
    print("FedSA-FTL ResNet Federated Learning")
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
    print(f"Privacy Method: Custom DP-SGD + {'Opacus RDP' if opacus_flag else 'Custom RDP'} accounting")
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
    
    # Prepare data
    print("\nPreparing federated data...")
    # Ensure ResNet-specific transforms are used
    config['data']['model_type'] = 'resnet'
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
        num_workers=config['data'].get('num_workers', 0)
    )
    
    # Initialize server
    print("Initializing federated server...")
    server = FedSAFTLServer(device)
    
    # Create model and get statistics
    print(f"Creating {config['model']['model_name']} model...")
    # Add seed to model config for reproducible LoRA initialization
    model_config = config['model'].copy()
    model_config['seed'] = config.get('seed', 42)
    temp_model = create_model_resnet(model_config)
    temp_client = ResNetFedSAFTLClient(0, temp_model, device)
    model_stats = temp_client.get_model_size()
    
    print("\nResNet Model Statistics:")
    print(f"  Architecture: {config['model']['model_name']}")
    print(f"  Total parameters: {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    print(f"  LoRA rank: {config['model']['lora_r']}")
    print(f"  Communication overhead: {model_stats['communication_params']:,} parameters/round")
    print(f"  Compression ratio: {model_stats['compression_ratio']:.2f}x")
    
    # Create clients
    print(f"\nCreating {config['federated']['num_clients']} ResNet federated clients...")
    clients = []
    
    
    # Create results directory with date and ResNet identifier
    current_date = datetime.now().strftime('%m%d')  # MMDD format
    base_experiment_dir = Path(config.get('experiment', {}).get('output_dir', 'experiments/quickstart_resnet'))
    experiment_dir = base_experiment_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure directory exists
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {experiment_dir}")
    
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
        
        # Add seed to model config for reproducible LoRA initialization
        model_config = config['model'].copy()
        model_config['seed'] = config.get('seed', 42)
        client_model = create_model_resnet(model_config)
        client = ResNetFedSAFTLClient(
            client_id, 
            client_model, 
            device,
            config,  # Pass config for DP and aggregation settings
            privacy_mechanism  # 各クライアントに固有のオブジェクトを渡す
        )
        clients.append(client)
    
    print("Starting ResNet federated training...")
    print("=" * 80)
    
    # Training loop
    best_accuracy = 0
    best_round = 0
    start_time = time.time()
    
    for round_idx in range(config['federated']['num_rounds']):
        print(f"\n[Round {round_idx + 1}/{config['federated']['num_rounds']}]")
        
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
        
        for client_id in selected_clients:
            # Get client training data with optional Mixup/CutMix
            client_dataloader = get_client_dataloader(
                trainset,
                client_train_indices[client_id],
                config['data']['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,  # Use Mixup/CutMix if enabled
                num_workers=config['data'].get('num_workers', 0)
            )
            
            # Update with global parameters based on aggregation method
            aggregation_method = config['federated'].get('aggregation_method', 'fedavg')
            if aggregation_method == 'fedsa_shareA_dp':
                # Get global A parameters for FedSA
                if hasattr(server, 'global_A_params') and server.global_A_params:
                    clients[client_id].update_model({'A_params': server.global_A_params})
            else:
                # Standard federated learning
                global_params = server.get_global_A_params()
                if global_params:
                    clients[client_id].update_model(global_params)
            
            # Local training
            client_result = clients[client_id].train(client_dataloader, config['training'])
            client_updates.append(client_result)
            train_accuracies.append(client_result['accuracy'])
            
            print(f"  Client {client_id}: Loss={client_result['loss']:.4f}, "
                  f"Accuracy={client_result['accuracy']:.2f}%")
        
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
            test_accuracies = [0] * len(selected_clients)  # Placeholder
            personalized_accuracies = [0] * len(selected_clients)
            avg_personalized_acc = 0
            # Create proper test results with both accuracy and loss keys
            client_test_results = [{'accuracy': 0, 'loss': 0} for _ in selected_clients]
        
        # Server aggregation based on method
        aggregation_method = config['federated'].get('aggregation_method', 'fedavg')
        
        if aggregation_method == 'fedsa_shareA_dp':
            # FedSA with DP: aggregate only A matrices
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
            
            # Calculate communication cost (only A matrices)
            total_A_params = sum(p.numel() for p in aggregated_A.values())
            communication_cost_mb = (total_A_params * 4) / (1024 * 1024)  # 4 bytes per float32
            
            # Verify only A parameters were uploaded
            for i, update in enumerate(client_updates):
                if 'upload_type' in update and update['upload_type'] != 'A_matrices_only':
                    print(f"WARNING: Client {i} uploaded unexpected parameters!")
            
            # Enhanced privacy information logging
            if config.get('privacy', {}).get('enable_privacy', False):
                privacy_analyses = [update.get('privacy_analysis', {}) for update in client_updates if 'privacy_analysis' in update]
                if privacy_analyses:
                    # Average privacy metrics across clients
                    avg_custom_eps = sum(analysis.get('custom_epsilon', 0) for analysis in privacy_analyses) / len(privacy_analyses)
                    
                    print(f"\n  === Privacy Analysis (A matrices only) ===")
                    print(f"  Custom RDP approximation: ε={avg_custom_eps:.4f}")
                    
                    # Show Opacus results if available
                    opacus_epsilons = [analysis.get('opacus_epsilon') for analysis in privacy_analyses if analysis.get('opacus_epsilon') != 'Not available']
                    if opacus_epsilons:
                        avg_opacus_eps = sum(opacus_epsilons) / len(opacus_epsilons)
                        print(f"  Opacus RDP accounting: ε={avg_opacus_eps:.4f} (recommended for academic use)")
                        print(f"  Privacy Budget: ε≤{config['privacy'].get('epsilon', 8.0)}")
                    else:
                        print(f"  Opacus accounting: Not available (install opacus for accurate ε)")
                    
                    # Show key privacy parameters
                    if privacy_analyses:
                        sample_analysis = privacy_analyses[0]
                        print(f"  Sampling ratio: {sample_analysis.get('sampling_ratio', 0):.4f}")
                        print(f"  Noise multiplier: {sample_analysis.get('noise_multiplier', 0)}")
                        print(f"  Steps taken: {sample_analysis.get('steps_taken', 0)}")
                        print(f"  Note: Privacy protection applied only to shared A matrices")
                    print(f"  =============================================")
            
            round_stats = {
                'communication_cost_mb': communication_cost_mb,
                'aggregated_params': total_A_params,
                'aggregation_method': 'fedsa_shareA_dp'
            }
        else:
            # Standard federated learning
            round_stats = server.federated_round(client_updates, client_test_results)
        
        # Print summary
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)
        
        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Avg Training Accuracy: {avg_train_acc:.2f}%")
        
        # Track best accuracy only during evaluation rounds
        is_new_best = False
        if is_eval_round and test_accuracies:
            print(f"  Avg Personalized Test Accuracy: {avg_personalized_acc:.2f}%")
            # Track best PERSONALIZED accuracy as the main metric
            if avg_personalized_acc > best_accuracy:
                best_accuracy = avg_personalized_acc
                best_round = round_idx + 1
                is_new_best = True
                print(f"  ** New best personalized accuracy! **")
        
        print(f"  Communication Cost: {round_stats.get('communication_cost_mb', 0):.2f} MB")
        
        # Save round results
        round_result = {
            'round': round_idx + 1,
            'timestamp': datetime.now().isoformat(),
            'selected_clients': selected_clients,
            'avg_train_accuracy': avg_train_acc,
            'avg_personalized_accuracy': avg_personalized_acc if any(test_accuracies) else 0,
            'individual_train_accuracies': train_accuracies,
            'individual_test_accuracies': test_accuracies if any(test_accuracies) else [],
            'communication_cost_mb': round_stats.get('communication_cost_mb', 0),
            'is_best_round': is_new_best
        }
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
                'test_accuracy': avg_personalized_acc,
                'communication_cost_mb': round_stats.get('communication_cost_mb', 0)
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
    results['summary'] = {
        'best_test_accuracy': best_accuracy,
        'best_round': best_round,
        'total_rounds': config['federated']['num_rounds'],
        'total_communication_mb': sum(server.history['communication_cost']),  # Already in MB
        'final_avg_accuracy': avg_personalized_acc if 'avg_personalized_acc' in locals() else 0,
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
            df_data.append({
                'round': round_data['round'],
                'train_accuracy': round_data['avg_train_accuracy'],
                'test_accuracy': round_data['avg_personalized_accuracy'],
                'communication_mb': round_data['communication_cost_mb'],
                'is_best': round_data['is_best_round']
            })
        
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
    print(f"Total Communication: {sum(server.history['communication_cost']):.2f} MB")
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
        
        summary_for_notification = {
            'final_avg_accuracy': avg_personalized_acc if 'avg_personalized_acc' in locals() else 0,
            'final_std_accuracy': 0,  # Can calculate from individual client results if needed
            'best_test_accuracy': best_accuracy,
            'total_rounds': config['federated']['num_rounds'],
            'total_communication_mb': sum(server.history['communication_cost'])  # Already in MB
        }
        
        slack_notifier.send_training_complete(
            config_for_notification,
            summary_for_notification,
            training_duration
        )
    
    # Evaluation
    if best_accuracy > 20:  # Reasonable threshold for CIFAR-100
        print("🎉 ResNet federated learning successful!")
    elif best_accuracy > 10:
        print("⚠️ ResNet learning in progress - consider more rounds")
    else:
        print("❌ ResNet learning may need hyperparameter tuning")


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