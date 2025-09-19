#!/usr/bin/env python3
"""
Standard FedAvg implementation for ResNet on CIFAR-100
Pure federated averaging without LoRA adapters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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

# Import from src modules
from data_utils import prepare_federated_data, get_client_dataloader, MixupCutmixCollate
from notification_utils import SlackNotifier
from cifar_resnet import create_cifar_resnet  # Import CIFAR-optimized ResNet


class StandardResNet(nn.Module):
    """CIFAR-optimized ResNet without LoRA for FedAvg"""
    
    def __init__(self, model_name='resnet18', num_classes=100, pretrained=False):
        super().__init__()
        
        # Use CIFAR-optimized ResNet (32x32 input, no maxpool)
        self.model = create_cifar_resnet(model_name, num_classes, pretrained=False)
        # Note: CIFAR ResNet doesn't use ImageNet pretrained weights
        
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        """Extract features before the final classifier"""
        return self.model.extract_features(x)


class FedAvgClient:
    """Standard FedAvg client implementation"""
    
    def __init__(self, client_id, model, device):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.local_data_size = 0
    
    def train(self, dataloader, training_config):
        """Train the model locally with AMP support"""
        self.model.train()
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=training_config.get('lr', 0.01),
            momentum=training_config.get('momentum', 0.9),
            weight_decay=training_config.get('weight_decay', 0.0005)
        )
        
        # Setup AMP scaler
        scaler = torch.cuda.amp.GradScaler()
        
        num_epochs = training_config.get('epochs', 3)
        total_loss = 0
        correct = 0
        total = 0
        
        self.local_data_size = len(dataloader.dataset)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Use autocast for forward pass
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    
                    # Handle Mixup/CutMix targets
                    if len(target.shape) > 1:  # Mixup/CutMix
                        loss = -(target * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, target)
                
                # Scale loss and backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping if specified (with scaled gradients)
                if training_config.get('gradient_clip'):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        training_config['gradient_clip']
                    )
                
                # Update weights with scaler
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy (only for non-Mixup targets)
                if len(target.shape) == 1:
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (num_epochs * len(dataloader))
        accuracy = 100. * correct / total if total > 0 else 0
        
        print(f"  Client {self.client_id}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        # Return model state and metrics
        return {
            'model_state': self.model.state_dict(),
            'num_samples': self.local_data_size,
            'local_data_size': self.local_data_size,
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def update_model(self, global_state):
        """Update model with global parameters"""
        if 'model_state' in global_state:
            self.model.load_state_dict(global_state['model_state'])
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Use autocast for forward pass in evaluation
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= total
        accuracy = 100. * correct / total
        
        return {
            'loss': test_loss,
            'accuracy': accuracy
        }
    
    def get_model_size(self):
        """Get model size information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'communication_params': total_params  # All parameters are communicated
        }


class FedAvgServer:
    """Standard FedAvg server implementation"""
    
    def __init__(self, device):
        self.device = device
        self.global_model_state = None
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'communication_cost': []
        }
    
    def aggregate(self, client_updates):
        """Aggregate client models using weighted averaging"""
        # Calculate weights based on number of samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        weights = [update['num_samples'] / total_samples for update in client_updates]
        
        # Initialize aggregated state
        aggregated_state = {}
        first_state = client_updates[0]['model_state']
        
        # Aggregate each parameter
        for param_name in first_state.keys():
            # Use float32 for aggregation to handle different dtypes
            aggregated_param = torch.zeros_like(first_state[param_name], dtype=torch.float32)
            
            for client_update, weight in zip(client_updates, weights):
                if param_name in client_update['model_state']:
                    param_data = client_update['model_state'][param_name].float()
                    aggregated_param += weight * param_data
            
            # Convert back to original dtype
            original_dtype = first_state[param_name].dtype
            aggregated_state[param_name] = aggregated_param.to(original_dtype)
        
        self.global_model_state = aggregated_state
        
        # Calculate metrics
        avg_loss = np.mean([update['loss'] for update in client_updates])
        avg_accuracy = np.mean([update['accuracy'] for update in client_updates])
        
        # Calculate communication cost
        total_params = sum(p.numel() for p in aggregated_state.values())
        communication_cost_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        
        # Update history
        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(avg_accuracy)
        self.history['communication_cost'].append(communication_cost_mb)
        
        return {
            'avg_train_loss': avg_loss,
            'avg_train_accuracy': avg_accuracy,
            'communication_cost_mb': communication_cost_mb,
            'total_params': total_params
        }
    
    def get_global_model_state(self):
        """Get global model state"""
        return {'model_state': self.global_model_state} if self.global_model_state else None


def create_model(config):
    """Create standard ResNet model"""
    return StandardResNet(
        model_name=config.get('model_name', 'resnet18'),
        num_classes=config.get('num_classes', 100),
        pretrained=config.get('pretrained', True)
    )


def main():
    parser = argparse.ArgumentParser(description='Standard FedAvg with ResNet')
    parser.add_argument('--config', type=str, default='configs/experiment_configs_iid/IID-FedAvg.yaml',
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
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
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
    print("Standard FedAvg with ResNet")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    print(f"Model: {config['model']['model_name']} (Standard ResNet without LoRA)")
    print(f"Dataset: {config['data']['dataset_name'].upper()}")
    print(f"Clients: {config['federated']['num_clients']}")
    print(f"Rounds: {config['federated']['num_rounds']}")
    print("=" * 80)
    
    # Set seed for reproducibility BEFORE model creation
    if 'seed' in config:
        seed = config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f"🔒 Random seed set to {seed} for reproducible model initialization")
        
        if config.get('reproducibility', {}).get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    
    # Prepare data
    print("\nPreparing federated data...")
    config['data']['model_type'] = 'resnet'
    config['data']['use_cifar_resnet'] = True  # Enable CIFAR-optimized transforms (32x32)
    trainset, testset, client_train_indices, client_test_indices = prepare_federated_data(config['data'])
    
    # Prepare Mixup/CutMix if enabled
    collate_fn = None
    augmentation_config = config['data'].get('augmentations', {})
    mixup_config = augmentation_config.get('mixup', {})
    cutmix_config = augmentation_config.get('cutmix', {})
    
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
    
    # Create initial global model 
    print(f"Creating initial global {config['model']['model_name']} model...")
    global_model = create_model(config['model'])
    
    # Initialize server with global model
    print("Initializing FedAvg server...")
    server = FedAvgServer(device)
    server.global_model_state = global_model.state_dict()  # Set initial global state
    
    # Create model for statistics
    temp_client = FedAvgClient(0, global_model, device)
    model_stats = temp_client.get_model_size()
    
    print("\nStandard ResNet Model Statistics:")
    print(f"  Architecture: {config['model']['model_name']}")
    print(f"  Total parameters: {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    print(f"  Communication overhead: {model_stats['communication_params']:,} parameters/round")
    print(f"  No LoRA adapters - Full model aggregation")
    print(f"  Initial model synchronized across all clients")
    
    # Initialize Slack notifier
    slack_notifier = None
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if webhook_url:
        slack_notifier = SlackNotifier(webhook_url)
        print("Slack notifications enabled")
    else:
        print("Slack notifications disabled (set SLACK_WEBHOOK_URL environment variable to enable)")
    
    # Create clients with synchronized initial model
    print(f"Creating {config['federated']['num_clients']} FedAvg clients...")
    clients = []
    
    for client_id in range(config['federated']['num_clients']):
        # Create model and load the same initial state as server
        client_model = create_model(config['model'])
        client_model.load_state_dict(server.global_model_state)  # Sync with initial global model
        client = FedAvgClient(client_id, client_model, device)
        clients.append(client)
    
    print(f"✅ All {len(clients)} clients initialized with synchronized global model")
    
    # Create experiment directory
    current_date = datetime.now().strftime('%m%d')
    base_experiment_dir = Path(config.get('experiment', {}).get('output_dir', 'experiments/fedavg_resnet'))
    experiment_dir = base_experiment_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {experiment_dir}")
    
    # File naming suffix
    date_suffix = f"{current_date}FedAvg"
    
    # Initialize results tracking
    results = {
        'config': config,
        'start_time': datetime.now().isoformat(),
        'experiment_dir': str(experiment_dir),
        'date_suffix': date_suffix,
        'rounds': []
    }
    
    print("Starting FedAvg training...")
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
            selected_clients = list(range(num_clients))
        else:
            selected_clients = sorted(random.sample(range(num_clients), num_selected))
        
        print(f"Selected clients: {selected_clients}")
        
        # Client updates
        client_updates = []
        
        for client_id in selected_clients:
            # Get client training data
            client_dataloader = get_client_dataloader(
                trainset,
                client_train_indices[client_id],
                config['data']['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=config['data'].get('num_workers', 0)
            )
            
            # Update with global model (always available since server is initialized with model)
            global_state = server.get_global_model_state()
            if global_state:
                clients[client_id].update_model(global_state)
            else:
                print(f"WARNING: No global model state available for client {client_id}")
            
            # Local training
            client_result = clients[client_id].train(client_dataloader, config['training'])
            client_updates.append(client_result)
        
        # Server aggregation
        round_stats = server.aggregate(client_updates)
        
        # Print summary
        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Avg Training Accuracy: {round_stats['avg_train_accuracy']:.2f}%")
        print(f"  Communication Cost: {round_stats['communication_cost_mb']:.2f} MB")
        
        # Evaluation
        is_eval_round = (round_idx + 1) % config['evaluation'].get('eval_freq', 5) == 0
        test_accuracies = []
        avg_test_acc = 0
        
        if is_eval_round:
            print(f"  Evaluating on test data...")
            for client_id in selected_clients:
                client_test_dataloader = get_client_dataloader(
                    testset,
                    client_test_indices[client_id],
                    config['data']['batch_size'],
                    shuffle=False,
                    num_workers=config['data'].get('num_workers', 0)
                )
                test_result = clients[client_id].evaluate(client_test_dataloader)
                test_accuracies.append(test_result['accuracy'])
                print(f"    Client {client_id}: {test_result['accuracy']:.2f}%")
            
            avg_test_acc = sum(test_accuracies) / len(test_accuracies)
            print(f"  Average Test Accuracy: {avg_test_acc:.2f}%")
            
            # Track best accuracy
            if avg_test_acc > best_accuracy:
                best_accuracy = avg_test_acc
                best_round = round_idx + 1
                print(f"  ** New best accuracy! **")
            
            server.history['test_accuracy'].append(avg_test_acc)
        else:
            test_accuracies = None  # No evaluation this round
            avg_test_acc = None  # No evaluation this round
        
        # Save round results
        round_result = {
            'round': round_idx + 1,
            'timestamp': datetime.now().isoformat(),
            'selected_clients': selected_clients,
            'avg_train_accuracy': round_stats['avg_train_accuracy'],
            'avg_test_accuracy': avg_test_acc if avg_test_acc is not None else None,
            'individual_test_accuracies': test_accuracies if test_accuracies is not None else None,
            'communication_cost_mb': round_stats['communication_cost_mb'],
            'is_best_round': avg_test_acc is not None and avg_test_acc == best_accuracy
        }
        results['rounds'].append(round_result)
        
        # Save results to file
        results_file = experiment_dir / f'training_results_{date_suffix}.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save results file: {e}")
        
        # Send progress notification
        if slack_notifier and (round_idx + 1) % 10 == 0:
            config_for_notification = {
                'experiment': {'name': f'FedAvg ({config["model"]["model_name"]})'},
                'federated': config['federated']
            }
            round_stats_for_notification = {
                'train_accuracy': round_stats['avg_train_accuracy'],
                'test_accuracy': avg_test_acc,
                'communication_cost_mb': round_stats['communication_cost_mb']
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
        
        # Checkpoint
        if (round_idx + 1) % config['federated'].get('checkpoint_freq', 20) == 0:
            print(f"  Checkpoint saved at round {round_idx + 1}")
            checkpoint_file = experiment_dir / f'checkpoint_round_{round_idx + 1}_{date_suffix}.json'
            checkpoint_data = {
                'round': round_idx + 1,
                'best_accuracy': best_accuracy,
                'best_round': best_round,
                'current_accuracy': avg_test_acc,
                'recent_rounds': results['rounds'][-10:]
            }
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Warning: Failed to save checkpoint: {e}")
    
    # Final results
    end_time = time.time()
    training_duration = end_time - start_time
    
    results['end_time'] = datetime.now().isoformat()
    results['training_duration_seconds'] = training_duration
    results['summary'] = {
        'best_test_accuracy': best_accuracy,
        'best_round': best_round,
        'total_rounds': config['federated']['num_rounds'],
        'total_communication_mb': sum(server.history['communication_cost']),
        'final_avg_accuracy': avg_test_acc,
        'training_duration_hours': training_duration / 3600,
        'model_name': config['model']['model_name'],
        'dataset': config['data']['dataset_name'],
        'aggregation_method': 'fedavg'
    }
    
    # Save final results
    final_results_file = experiment_dir / f'final_results_{date_suffix}.json'
    try:
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {final_results_file}")
    except Exception as e:
        print(f"Warning: Failed to save final results: {e}")
    
    # Create summary CSV
    try:
        import pandas as pd
        df_data = []
        for round_data in results['rounds']:
            df_data.append({
                'round': round_data['round'],
                'train_accuracy': round_data['avg_train_accuracy'],
                'test_accuracy': round_data['avg_test_accuracy'],
                'communication_mb': round_data['communication_cost_mb'],
                'is_best': round_data['is_best_round']
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            csv_file = experiment_dir / f'results_summary_{date_suffix}.csv'
            df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved to: {csv_file}")
        
    except ImportError:
        print("Warning: pandas not available, skipping CSV export")
    except Exception as e:
        print(f"Warning: Failed to create CSV summary: {e}")
    
    print("\n" + "=" * 80)
    print("Standard FedAvg Training Complete!")
    print("=" * 80)
    print(f"Configuration: {config_path.name}")
    print(f"Model: {config['model']['model_name']} (Standard ResNet)")
    print(f"Best Test Accuracy: {best_accuracy:.2f}% (Round {best_round})")
    print(f"Total Rounds: {config['federated']['num_rounds']}")
    print(f"Total Communication: {sum(server.history['communication_cost']):.2f} MB")
    print(f"Training Duration: {training_duration / 3600:.2f} hours")
    print(f"Results saved to: {experiment_dir}")
    print(f"Aggregation Method: Standard FedAvg (Full Model)")
    print("=" * 80)
    
    # Send completion notification
    if slack_notifier:
        config_for_notification = {
            'experiment': {
                'name': f'FedAvg ({config["model"]["model_name"]})',
                'output_dir': str(experiment_dir)
            },
            'federated': config['federated']
        }
        summary_for_notification = {
            'final_avg_accuracy': avg_test_acc,
            'final_std_accuracy': 0,
            'best_test_accuracy': best_accuracy,
            'total_rounds': config['federated']['num_rounds'],
            'total_communication_mb': sum(server.history['communication_cost'])
        }
        slack_notifier.send_training_complete(
            config_for_notification,
            summary_for_notification,
            training_duration
        )
    
    # Evaluation
    if best_accuracy > 20:  # Reasonable threshold for CIFAR-100
        print("🎉 FedAvg training successful!")
    elif best_accuracy > 10:
        print("⚠️ FedAvg learning in progress - consider more rounds")
    else:
        print("❌ FedAvg learning may need hyperparameter tuning")


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
                    'experiment': {'name': 'FedAvg ResNet (ERROR)'}
                }
                slack_notifier.send_error_notification(
                    config_for_notification,
                    str(e) + "\n" + traceback.format_exc()[:500]
                )
            except Exception:
                pass