#!/usr/bin/env python3
"""
Quick start script for Vision Transformer (ViT) federated learning
Similar to quickstart_vgg16.py but using ViT models
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
from fedsa_ftl_model_vit import create_model_vit
from fedsa_ftl_client import FedSAFTLClient
from fedsa_ftl_server import FedSAFTLServer
from data_utils import prepare_federated_data, get_client_dataloader
from privacy_utils import DifferentialPrivacy
from notification_utils import SlackNotifier


class ViTFedSAFTLClient(FedSAFTLClient):
    """Extended client for ViT models"""
    
    def __init__(self, client_id, model, device, privacy_mechanism=None):
        super().__init__(client_id, model, device)
        # privacy_mechanismをセッターで設定
        if privacy_mechanism is not None:
            self.set_privacy_mechanism(privacy_mechanism)


def main():
    # Initialize Slack notifier
    slack_notifier = None
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if webhook_url:
        slack_notifier = SlackNotifier(webhook_url)
        print("Slack notifications enabled")
    else:
        print("Slack notifications disabled (set SLACK_WEBHOOK_URL environment variable to enable)")

    parser = argparse.ArgumentParser(description='FedSA-FTL ViT Quick Start')
    parser.add_argument('--config', type=str, default='configs/cifar100_vit_base.yaml',
                       help='Path to configuration file')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Override number of rounds')
    parser.add_argument('--clients', type=int, default=None,
                       help='Override number of clients')
    parser.add_argument('--model', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], 
                       default=None, help='Override ViT model variant')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Available configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*vit*.yaml"):
                print(f"  {config_file}")
        return
    
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
    
    # Set seed
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
    
    # Prepare data
    print("\nPreparing federated data...")
    # Ensure ViT-specific transforms are used
    config['data']['model_type'] = 'vit'
    trainset, testset, client_train_indices, client_test_indices = prepare_federated_data(config['data'])
    
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
    temp_model = create_model_vit(config['model'])
    temp_client = ViTFedSAFTLClient(0, temp_model, device)
    model_stats = temp_client.get_model_size()
    
    print("\nViT Model Statistics:")
    print(f"  Architecture: {config['model']['model_name']}")
    print(f"  Total parameters: {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    print(f"  LoRA rank: {config['model']['lora_r']}")
    print(f"  Communication overhead: {model_stats['communication_params']:,} parameters/round")
    print(f"  Compression ratio: {model_stats['compression_ratio']:.2f}x")
    
    # Create clients
    print(f"\nCreating {config['federated']['num_clients']} ViT federated clients...")
    clients = []
    
    
    # Create results directory with date and ViT identifier
    current_date = datetime.now().strftime('%m%d')  # MMDD format
    base_experiment_dir = Path(config.get('experiment', {}).get('output_dir', 'experiments/quickstart_vit'))
    experiment_dir = base_experiment_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure directory exists
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {experiment_dir}")
    
    # Generate file names with date and ViT identifier
    date_vit_suffix = f"{current_date}ViT"
    
    # Initialize results tracking
    results = {
        'config': config,
        'start_time': datetime.now().isoformat(),
        'experiment_dir': str(experiment_dir),
        'date_suffix': date_vit_suffix,
        'rounds': [],
        'summary': {}
    }
    
    # Send training start notification
    if slack_notifier:
        config_for_notification = {
            'experiment': {'name': f'ViT QuickStart ({config["model"]["model_name"]})'},
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
        
        client_model = create_model_vit(config['model'])
        client = ViTFedSAFTLClient(
            client_id, 
            client_model, 
            device,
            privacy_mechanism  # 各クライアントに固有のオブジェクトを渡す
        )
        clients.append(client)
    
    print("Starting ViT federated training...")
    print("=" * 80)
    
    # Training loop
    best_accuracy = 0
    best_round = 0
    start_time = time.time()
    
    for round_idx in range(config['federated']['num_rounds']):
        print(f"\n[Round {round_idx + 1}/{config['federated']['num_rounds']}]")
        
        # Select ALL clients for participation (full client participation)
        # This ensures all clients participate in every round for accurate personalized evaluation
        selected_clients = list(range(config['federated']['num_clients']))
        
        print(f"Selected clients: {selected_clients}")
        
        # Client updates
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
        
        # Evaluation
        if (round_idx + 1) % config.get('evaluation', {}).get('eval_freq', 5) == 0:
            print("\nEvaluating models...")
            
            # 1. Personalized Accuracy: Each client evaluates on their LOCAL test data
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
        
        # Server aggregation
        round_stats = server.federated_round(client_updates, client_test_results)
        
        # Print summary
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)
        avg_test_acc = sum(test_accuracies) / len(test_accuracies) if any(test_accuracies) else 0
        
        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Avg Training Accuracy: {avg_train_acc:.2f}%")
        if any(test_accuracies):
            print(f"  Avg Personalized Test Accuracy: {avg_personalized_acc:.2f}%")
            # Track best PERSONALIZED accuracy as the main metric
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
            'avg_test_accuracy': avg_test_acc,
            'avg_personalized_accuracy': avg_personalized_acc if any(test_accuracies) else 0,
            'individual_train_accuracies': train_accuracies,
            'individual_test_accuracies': test_accuracies if any(test_accuracies) else [],
            'individual_personalized_accuracies': personalized_accuracies if 'personalized_accuracies' in locals() and any(test_accuracies) else [],
            'communication_cost_mb': round_stats.get('communication_cost_mb', 0),
            'is_best_round': avg_personalized_acc > best_accuracy if any(test_accuracies) else False
        }
        results['rounds'].append(round_result)
        
        # Save results to file after each round (with date+ViT suffix)
        results_file = experiment_dir / f'training_results_{date_vit_suffix}.json'
        
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
                'experiment': {'name': f'ViT QuickStart ({config["model"]["model_name"]})'},
                'federated': config['federated']
            }
            
            # Prepare round stats for notification
            round_stats_for_notification = {
                'train_accuracy': avg_train_acc,
                'test_accuracy': avg_test_acc,
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
            # Save checkpoint results to separate file (with date+ViT suffix)
            checkpoint_file = experiment_dir / f'checkpoint_round_{round_idx + 1}_{date_vit_suffix}.json'
            checkpoint_data = {
                'round': round_idx + 1,
                'best_accuracy': best_accuracy,
                'best_round': best_round,
                'current_accuracy': avg_test_acc,
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
        'total_communication_mb': sum(server.history['communication_cost']) / (1024 * 1024),
        'final_avg_accuracy': avg_test_acc,
        'training_duration_hours': training_duration / 3600,
        'model_name': config['model']['model_name'],
        'dataset': config['data']['dataset_name']
    }
    
    # Save final results (with date+ViT suffix)
    final_results_file = experiment_dir / f'final_results_{date_vit_suffix}.json'
    
    try:
        final_results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {final_results_file}")
    except Exception as e:
        print(f"Warning: Failed to save final results: {e}")
    
    # Create summary CSV for easy plotting (with date+ViT suffix)
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
        
        if df_data:  # Only create CSV if we have data
            df = pd.DataFrame(df_data)
            csv_file = experiment_dir / f'results_summary_{date_vit_suffix}.csv'
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
    print("ViT Federated Learning Complete!")
    print("=" * 80)
    print(f"Configuration: {config_path.name}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Best Personalized Accuracy: {best_accuracy:.2f}% (Round {best_round})")
    print(f"Total Rounds: {config['federated']['num_rounds']}")
    print(f"Total Communication: {sum(server.history['communication_cost']) / (1024 * 1024):.2f} MB")
    print(f"Training Duration: {training_duration / 3600:.2f} hours")
    print(f"Results saved to: {experiment_dir}")
    print(f"  - Training results: final_results_{date_vit_suffix}.json")
    if csv_file:
        print(f"  - Summary CSV: results_summary_{date_vit_suffix}.csv")
    print(f"  - File naming pattern: *_{date_vit_suffix}.*")
    print("=" * 80)
    
    # Send completion notification
    if slack_notifier:
        config_for_notification = {
            'experiment': {
                'name': f'ViT QuickStart ({config["model"]["model_name"]})',
                'output_dir': str(experiment_dir)
            },
            'federated': config['federated'],
            'privacy': config.get('privacy', {})
        }
        
        summary_for_notification = {
            'final_avg_accuracy': avg_test_acc,
            'final_std_accuracy': 0,  # Can calculate from individual client results if needed
            'best_test_accuracy': best_accuracy,
            'total_rounds': config['federated']['num_rounds'],
            'total_communication_mb': sum(server.history['communication_cost']) / (1024 * 1024)
        }
        
        slack_notifier.send_training_complete(
            config_for_notification,
            summary_for_notification,
            training_duration
        )
    
    # Evaluation
    if best_accuracy > 20:  # Reasonable threshold for CIFAR-100
        print("🎉 ViT federated learning successful!")
    elif best_accuracy > 10:
        print("⚠️ ViT learning in progress - consider more rounds")
    else:
        print("❌ ViT learning may need hyperparameter tuning")


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
                    'experiment': {'name': 'ViT QuickStart (ERROR)'}
                }
                slack_notifier.send_error_notification(
                    config_for_notification,
                    str(e) + "\n" + traceback.format_exc()[:500]
                )
            except Exception:
                pass  # Don't fail on notification failure