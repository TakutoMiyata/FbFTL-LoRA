#!/usr/bin/env python3
"""
Refactored Vision Transformer (ViT) federated learning script
Improved organization, modularity, and maintainability
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
from typing import Dict, List, Tuple, Optional

# ====================== Environment Setup ======================

def load_env_file(env_path: str = '.env') -> None:
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

# Load environment variables
load_env_file()

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules after path setup
from fedsa_ftl_model_vit import create_model_vit
from fedsa_ftl_client import FedSAFTLClient
from fedsa_ftl_server import FedSAFTLServer
from data_utils import prepare_federated_data, get_client_dataloader
from privacy_utils import DifferentialPrivacy
from notification_utils import SlackNotifier


# ====================== Client Implementation ======================

class ViTFedSAFTLClient(FedSAFTLClient):
    """Extended client for ViT models"""
    
    def __init__(self, client_id: int, model, device: torch.device, 
                 privacy_mechanism: Optional[DifferentialPrivacy] = None):
        super().__init__(client_id, model, device)
        if privacy_mechanism is not None:
            self.set_privacy_mechanism(privacy_mechanism)


# ====================== Configuration Functions ======================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FedSA-FTL ViT Quick Start')
    parser.add_argument('--config', type=str, default='configs/cifar100_vit_base.yaml',
                       help='Path to configuration file')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Override number of rounds')
    parser.add_argument('--clients', type=int, default=None,
                       help='Override number of clients')
    parser.add_argument('--model', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], 
                       default=None, help='Override ViT model variant')
    return parser.parse_args()


def load_and_validate_config(args: argparse.Namespace) -> Tuple[Dict, Path]:
    """Load and validate configuration file"""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Available configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*vit*.yaml"):
                print(f"  {config_file}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command line overrides
    if args.rounds:
        config['federated']['num_rounds'] = args.rounds
    if args.clients:
        config['federated']['num_clients'] = args.clients
        config['data']['num_clients'] = args.clients
    if args.model:
        config['model']['model_name'] = args.model
    
    # Ensure ViT-specific settings
    config['data']['model_type'] = 'vit'
    
    return config, config_path


# ====================== Initialization Functions ======================

def initialize_environment(config: Dict) -> torch.device:
    """Initialize random seeds and device"""
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
    
    device = torch.device('cuda' if config.get('use_gpu', False) and torch.cuda.is_available() else 'cpu')
    return device


def create_experiment_directory(config: Dict) -> Tuple[Path, str]:
    """Create experiment directory and generate naming suffix"""
    current_date = datetime.now().strftime('%m%d')
    base_dir = Path(config.get('experiment', {}).get('output_dir', 'experiments/quickstart_vit'))
    experiment_dir = base_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    date_suffix = f"{current_date}ViT"
    return experiment_dir, date_suffix


def initialize_clients(config: Dict, device: torch.device) -> List[ViTFedSAFTLClient]:
    """Initialize all federated learning clients"""
    clients = []
    privacy_enabled = config.get('privacy', {}).get('enable_privacy', False)
    
    for client_id in range(config['federated']['num_clients']):
        privacy_mechanism = None
        if privacy_enabled:
            privacy_mechanism = DifferentialPrivacy(
                epsilon=config['privacy'].get('epsilon', 10.0),
                delta=config['privacy'].get('delta', 1e-5),
                max_grad_norm=config['privacy'].get('max_grad_norm', 0.5),
                total_rounds=config['federated'].get('num_rounds', 100)
            )
        
        client_model = create_model_vit(config['model'])
        client = ViTFedSAFTLClient(
            client_id, 
            client_model, 
            device,
            privacy_mechanism
        )
        clients.append(client)
    
    return clients


# ====================== Training Functions ======================

def perform_client_training(clients: List[ViTFedSAFTLClient], 
                          selected_clients: List[int],
                          server: FedSAFTLServer,
                          trainset, 
                          client_train_indices: List[List[int]],
                          config: Dict) -> Tuple[List[Dict], List[float]]:
    """Perform local training for selected clients"""
    client_updates = []
    train_accuracies = []
    
    for client_id in selected_clients:
        # Get client training data
        client_dataloader = get_client_dataloader(
            trainset,
            client_train_indices[client_id],
            config['data']['batch_size'],
            shuffle=True
        )
        
        # Update with global parameters
        global_A_params = server.get_global_A_params()
        if global_A_params:
            clients[client_id].update_model(global_A_params)
        
        # Local training
        client_result = clients[client_id].train(client_dataloader, config['training'])
        client_updates.append(client_result)
        train_accuracies.append(client_result['accuracy'])
        
        print(f"  Client {client_id}: Loss={client_result['loss']:.4f}, "
              f"Accuracy={client_result['accuracy']:.2f}%")
    
    return client_updates, train_accuracies


def evaluate_personalized_models(clients: List[ViTFedSAFTLClient],
                                selected_clients: List[int],
                                testset,
                                client_test_indices: List[List[int]],
                                config: Dict) -> Tuple[List[Dict], List[float], float]:
    """Evaluate personalized models on local test data"""
    print("\nEvaluating models...")
    print("  Personalized Accuracy (local test data):")
    
    personalized_results = []
    personalized_accuracies = []
    
    for client_id in selected_clients:
        # Get client's LOCAL test data
        client_test_dataloader = get_client_dataloader(
            testset,
            client_test_indices[client_id],
            config['data']['batch_size'],
            shuffle=False
        )
        
        # Evaluate personalized model on local test data
        test_result = clients[client_id].evaluate(client_test_dataloader)
        personalized_results.append(test_result)
        personalized_accuracies.append(test_result['accuracy'])
        print(f"    Client {client_id}: {test_result['accuracy']:.2f}%")
    
    avg_personalized_acc = sum(personalized_accuracies) / len(personalized_accuracies)
    print(f"  Average Personalized Accuracy: {avg_personalized_acc:.2f}%")
    
    return personalized_results, personalized_accuracies, avg_personalized_acc


# ====================== Result Management Functions ======================

def save_results(results: Dict, experiment_dir: Path, date_suffix: str) -> None:
    """Save results to JSON file"""
    results_file = experiment_dir / f'training_results_{date_suffix}.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save results file: {e}")


def save_checkpoint(round_idx: int, best_accuracy: float, best_round: int,
                   current_accuracy: float, results: Dict,
                   experiment_dir: Path, date_suffix: str) -> None:
    """Save checkpoint at specified intervals"""
    checkpoint_file = experiment_dir / f'checkpoint_round_{round_idx + 1}_{date_suffix}.json'
    checkpoint_data = {
        'round': round_idx + 1,
        'best_accuracy': best_accuracy,
        'best_round': best_round,
        'current_accuracy': current_accuracy,
        'recent_rounds': results['rounds'][-10:]
    }
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        print(f"  Checkpoint saved to: {checkpoint_file}")
    except Exception as e:
        print(f"  Warning: Failed to save checkpoint: {e}")


def create_summary_csv(results: Dict, experiment_dir: Path, date_suffix: str) -> Optional[Path]:
    """Create CSV summary for visualization"""
    try:
        import pandas as pd
        df_data = []
        for round_data in results['rounds']:
            df_data.append({
                'round': round_data['round'],
                'train_accuracy': round_data['avg_train_accuracy'],
                'personalized_accuracy': round_data['avg_personalized_accuracy'],
                'communication_mb': round_data['communication_cost_mb'],
                'is_best': round_data['is_best_round']
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            csv_file = experiment_dir / f'results_summary_{date_suffix}.csv'
            df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved to: {csv_file}")
            return csv_file
            
    except ImportError:
        print("Warning: pandas not available, skipping CSV export")
    except Exception as e:
        print(f"Warning: Failed to create CSV summary: {e}")
    
    return None


# ====================== Display Functions ======================

def print_configuration_summary(config: Dict, config_path: Path, device: torch.device) -> None:
    """Print configuration summary"""
    print("=" * 80)
    print("FedSA-FTL Vision Transformer (ViT) Federated Learning")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Dataset: {config['data']['dataset_name'].upper()}")
    print(f"Clients: {config['federated']['num_clients']}")
    print(f"Rounds: {config['federated']['num_rounds']}")
    print(f"Privacy: {'Enabled' if config.get('privacy', {}).get('enable_privacy', False) else 'Disabled'}")
    print("=" * 80)


def print_model_statistics(config: Dict, model_stats: Dict) -> None:
    """Print model statistics"""
    print("\nViT Model Statistics:")
    print(f"  Architecture: {config['model']['model_name']}")
    print(f"  Total parameters: {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    print(f"  LoRA rank: {config['model']['lora_r']}")
    print(f"  Communication overhead: {model_stats['communication_params']:,} parameters/round")
    print(f"  Compression ratio: {model_stats['compression_ratio']:.2f}x")


def print_final_summary(config: Dict, config_path: Path, best_accuracy: float,
                       best_round: int, training_duration: float,
                       server: FedSAFTLServer, experiment_dir: Path,
                       date_suffix: str, csv_file: Optional[Path]) -> None:
    """Print final training summary"""
    print("\n" + "=" * 80)
    print("ViT Federated Learning Complete!")
    print("=" * 80)
    print(f"Configuration: {config_path.name}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Best Personalized Accuracy: {best_accuracy:.2f}% (Round {best_round})")
    print(f"Total Rounds: {config['federated']['num_rounds']}")
    print(f"Total Communication: {sum(server.history['communication_cost']) / (1024 * 1024):.2f} MB")
    print(f"Training Duration: {training_duration / 3600:.2f} hours")
    print(f"Results saved to: {experiment_dir}")
    print(f"  - Training results: training_results_{date_suffix}.json")
    print(f"  - Final results: final_results_{date_suffix}.json")
    if csv_file:
        print(f"  - Summary CSV: {csv_file.name}")
    print(f"  - File naming pattern: *_{date_suffix}.*")
    print("=" * 80)
    
    # Performance evaluation
    if best_accuracy > 20:
        print("🎉 ViT federated learning successful!")
    elif best_accuracy > 10:
        print("⚠️ ViT learning in progress - consider more rounds")
    else:
        print("❌ ViT learning may need hyperparameter tuning")


# ====================== Main Training Loop ======================

def federated_training_loop(config: Dict, clients: List[ViTFedSAFTLClient],
                          server: FedSAFTLServer, trainset, testset,
                          client_train_indices: List[List[int]],
                          client_test_indices: List[List[int]],
                          experiment_dir: Path, date_suffix: str,
                          slack_notifier: Optional[SlackNotifier]) -> Dict:
    """Main federated training loop"""
    # Initialize results tracking
    results = {
        'config': config,
        'start_time': datetime.now().isoformat(),
        'experiment_dir': str(experiment_dir),
        'date_suffix': date_suffix,
        'rounds': [],
        'summary': {}
    }
    
    best_accuracy = 0
    best_round = 0
    start_time = time.time()
    
    print("Starting ViT federated training...")
    print("=" * 80)
    
    for round_idx in range(config['federated']['num_rounds']):
        print(f"\n[Round {round_idx + 1}/{config['federated']['num_rounds']}]")
        
        # Select all clients (full participation)
        selected_clients = list(range(config['federated']['num_clients']))
        print(f"Selected clients: {selected_clients}")
        
        # Client training phase
        client_updates, train_accuracies = perform_client_training(
            clients, selected_clients, server, trainset, client_train_indices, config
        )
        
        # Evaluation phase
        if (round_idx + 1) % config.get('evaluation', {}).get('eval_freq', 5) == 0:
            client_test_results, test_accuracies, avg_personalized_acc = evaluate_personalized_models(
                clients, selected_clients, testset, client_test_indices, config
            )
        else:
            test_accuracies = []
            avg_personalized_acc = 0
            client_test_results = [{'accuracy': 0, 'loss': 0} for _ in selected_clients]
        
        # Server aggregation
        round_stats = server.federated_round(client_updates, client_test_results)
        
        # Calculate and print round summary
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)
        
        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Avg Training Accuracy: {avg_train_acc:.2f}%")
        if test_accuracies:
            print(f"  Avg Personalized Test Accuracy: {avg_personalized_acc:.2f}%")
            if avg_personalized_acc > best_accuracy:
                best_accuracy = avg_personalized_acc
                best_round = round_idx + 1
                print(f"  ** New best personalized accuracy! **")
        print(f"  Communication Cost: {round_stats.get('communication_cost_mb', 0):.2f} MB")
        
        # Save round results
        round_result = {
            'round': round_idx + 1,
            'timestamp': datetime.now().isoformat(),
            'selected_clients': selected_clients,
            'avg_train_accuracy': avg_train_acc,
            'avg_personalized_accuracy': avg_personalized_acc,
            'individual_train_accuracies': train_accuracies,
            'individual_test_accuracies': test_accuracies,
            'communication_cost_mb': round_stats.get('communication_cost_mb', 0),
            'is_best_round': avg_personalized_acc > best_accuracy if test_accuracies else False
        }
        results['rounds'].append(round_result)
        
        # Save results after each round
        save_results(results, experiment_dir, date_suffix)
        
        # Send Slack notification
        if slack_notifier and (round_idx + 1) % 10 == 0:
            send_progress_notification(slack_notifier, config, round_idx, 
                                      avg_train_acc, avg_personalized_acc,
                                      round_stats, best_accuracy)
        
        # Save checkpoint
        if (round_idx + 1) % config['federated'].get('checkpoint_freq', 20) == 0:
            print(f"  Saving checkpoint at round {round_idx + 1}")
            save_checkpoint(round_idx, best_accuracy, best_round, 
                          avg_personalized_acc, results, experiment_dir, date_suffix)
    
    # Calculate final statistics
    end_time = time.time()
    training_duration = end_time - start_time
    
    results['end_time'] = datetime.now().isoformat()
    results['training_duration_seconds'] = training_duration
    results['summary'] = {
        'best_test_accuracy': best_accuracy,
        'best_round': best_round,
        'total_rounds': config['federated']['num_rounds'],
        'total_communication_mb': sum(server.history['communication_cost']) / (1024 * 1024),
        'final_avg_accuracy': avg_personalized_acc,
        'training_duration_hours': training_duration / 3600,
        'model_name': config['model']['model_name'],
        'dataset': config['data']['dataset_name']
    }
    
    return results, best_accuracy, best_round, training_duration


# ====================== Notification Functions ======================

def initialize_slack_notifier() -> Optional[SlackNotifier]:
    """Initialize Slack notifier if configured"""
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if webhook_url:
        print("Slack notifications enabled")
        return SlackNotifier(webhook_url)
    else:
        print("Slack notifications disabled (set SLACK_WEBHOOK_URL environment variable to enable)")
        return None


def send_training_start_notification(slack_notifier: SlackNotifier, config: Dict) -> None:
    """Send training start notification"""
    config_for_notification = {
        'experiment': {'name': f'ViT QuickStart ({config["model"]["model_name"]})'},
        'federated': config['federated'],
        'privacy': config.get('privacy', {})
    }
    slack_notifier.send_training_start(config_for_notification)


def send_progress_notification(slack_notifier: SlackNotifier, config: Dict,
                              round_idx: int, avg_train_acc: float,
                              avg_personalized_acc: float, round_stats: Dict,
                              best_accuracy: float) -> None:
    """Send progress update notification"""
    config_for_notification = {
        'experiment': {'name': f'ViT QuickStart ({config["model"]["model_name"]})'},
        'federated': config['federated']
    }
    
    round_stats_for_notification = {
        'train_accuracy': avg_train_acc,
        'test_accuracy': avg_personalized_acc,
        'communication_cost_mb': round_stats.get('communication_cost_mb', 0)
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


def send_completion_notification(slack_notifier: SlackNotifier, config: Dict,
                                avg_personalized_acc: float, best_accuracy: float,
                                server: FedSAFTLServer, experiment_dir: Path,
                                training_duration: float) -> None:
    """Send training completion notification"""
    config_for_notification = {
        'experiment': {
            'name': f'ViT QuickStart ({config["model"]["model_name"]})',
            'output_dir': str(experiment_dir)
        },
        'federated': config['federated'],
        'privacy': config.get('privacy', {})
    }
    
    summary_for_notification = {
        'final_avg_accuracy': avg_personalized_acc,
        'final_std_accuracy': 0,
        'best_test_accuracy': best_accuracy,
        'total_rounds': config['federated']['num_rounds'],
        'total_communication_mb': sum(server.history['communication_cost']) / (1024 * 1024)
    }
    
    slack_notifier.send_training_complete(
        config_for_notification,
        summary_for_notification,
        training_duration
    )


# ====================== Main Function ======================

def main():
    """Main entry point for the federated learning experiment"""
    # Parse arguments and load configuration
    args = parse_arguments()
    config, config_path = load_and_validate_config(args)
    
    # Initialize environment
    device = initialize_environment(config)
    slack_notifier = initialize_slack_notifier()
    
    # Print configuration summary
    print_configuration_summary(config, config_path, device)
    
    # Prepare data
    print("\nPreparing federated data...")
    trainset, testset, client_train_indices, client_test_indices = prepare_federated_data(config['data'])
    
    # Initialize server
    print("Initializing federated server...")
    server = FedSAFTLServer(device)
    
    # Create and analyze model
    print(f"Creating {config['model']['model_name']} model...")
    temp_model = create_model_vit(config['model'])
    temp_client = ViTFedSAFTLClient(0, temp_model, device)
    model_stats = temp_client.get_model_size()
    print_model_statistics(config, model_stats)
    
    # Initialize clients
    print(f"\nCreating {config['federated']['num_clients']} ViT federated clients...")
    clients = initialize_clients(config, device)
    
    # Create experiment directory
    experiment_dir, date_suffix = create_experiment_directory(config)
    print(f"Results will be saved to: {experiment_dir}")
    
    # Send training start notification
    if slack_notifier:
        send_training_start_notification(slack_notifier, config)
    
    # Run federated training
    results, best_accuracy, best_round, training_duration = federated_training_loop(
        config, clients, server, trainset, testset,
        client_train_indices, client_test_indices,
        experiment_dir, date_suffix, slack_notifier
    )
    
    # Save final results
    final_results_file = experiment_dir / f'final_results_{date_suffix}.json'
    try:
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {final_results_file}")
    except Exception as e:
        print(f"Warning: Failed to save final results: {e}")
    
    # Create summary CSV
    csv_file = create_summary_csv(results, experiment_dir, date_suffix)
    
    # Print final summary
    print_final_summary(config, config_path, best_accuracy, best_round,
                       training_duration, server, experiment_dir,
                       date_suffix, csv_file)
    
    # Send completion notification
    if slack_notifier:
        avg_personalized_acc = results['summary']['final_avg_accuracy']
        send_completion_notification(slack_notifier, config, avg_personalized_acc,
                                   best_accuracy, server, experiment_dir,
                                   training_duration)


# ====================== Entry Point ======================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error notification
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        if webhook_url:
            try:
                slack_notifier = SlackNotifier(webhook_url)
                config_for_notification = {
                    'experiment': {'name': 'ViT QuickStart (ERROR)'}
                }
                slack_notifier.send_error_notification(
                    config_for_notification,
                    str(e) + "\n" + traceback.format_exc()[:500]
                )
            except Exception:
                pass