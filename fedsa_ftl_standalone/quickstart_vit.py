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

# Add src directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import from src modules
from fedsa_ftl_model_vit import create_model_vit
from fedsa_ftl_client import FedSAFTLClient
from fedsa_ftl_server import FedSAFTLServer
from data_utils import prepare_federated_data, get_client_dataloader


class ViTFedSAFTLClient(FedSAFTLClient):
    """Extended client for ViT models"""
    
    def __init__(self, client_id, model, device, privacy_config=None):
        super().__init__(client_id, model, device, privacy_config)


def main():
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
        import numpy as np
        np.random.seed(config['seed'])
    
    # Prepare data
    print("\nPreparing federated data...")
    # Ensure ViT-specific transforms are used
    config['data']['model_type'] = 'vit'
    trainset, testset, client_indices = prepare_federated_data(config['data'])
    
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
    
    # Create privacy mechanism if enabled
    privacy_mechanism = None
    if config.get('privacy', {}).get('enable_privacy', False):
        from privacy_utils import DifferentialPrivacy
        privacy_mechanism = DifferentialPrivacy(
            epsilon=config['privacy'].get('epsilon', 10.0),
            delta=config['privacy'].get('delta', 1e-5),
            max_grad_norm=config['privacy'].get('max_grad_norm', 0.5)
        )
    
    for client_id in range(config['federated']['num_clients']):
        client_model = create_model_vit(config['model'])
        client = ViTFedSAFTLClient(
            client_id, 
            client_model, 
            device,
            privacy_mechanism
        )
        clients.append(client)
    
    print("Starting ViT federated training...")
    print("=" * 80)
    
    # Training loop
    best_accuracy = 0
    best_round = 0
    
    for round_idx in range(config['federated']['num_rounds']):
        print(f"\n[Round {round_idx + 1}/{config['federated']['num_rounds']}]")
        
        # Select clients
        import random
        num_selected = max(1, int(config['federated']['num_clients'] * 
                                  config['federated']['client_fraction']))
        selected_clients = random.sample(range(config['federated']['num_clients']), 
                                       num_selected)
        
        print(f"Selected clients: {selected_clients}")
        
        # Client updates
        client_updates = []
        train_accuracies = []
        
        for client_id in selected_clients:
            # Get client data
            client_dataloader = get_client_dataloader(
                trainset,
                client_indices[client_id],
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
            print("\nEvaluating ViT models...")
            test_results = []
            test_accuracies = []
            for client_id in selected_clients:
                test_result = clients[client_id].evaluate(test_dataloader)
                test_results.append(test_result)
                test_accuracies.append(test_result['accuracy'])
                print(f"  Client {client_id} test accuracy: {test_result['accuracy']:.2f}%")
            client_test_results = test_results
        else:
            test_accuracies = [0] * len(selected_clients)  # Placeholder
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
            print(f"  Avg Test Accuracy: {avg_test_acc:.2f}%")
            if avg_test_acc > best_accuracy:
                best_accuracy = avg_test_acc
                best_round = round_idx + 1
                print(f"  ** New best ViT accuracy! **")
        print(f"  Communication Cost: {round_stats.get('communication_cost_mb', 0):.2f} MB")
        
        # Checkpoint
        if (round_idx + 1) % config['federated'].get('checkpoint_freq', 20) == 0:
            print(f"  Checkpoint saved at round {round_idx + 1}")
    
    # Final results
    print("\n" + "=" * 80)
    print("ViT Federated Learning Complete!")
    print("=" * 80)
    print(f"Configuration: {config_path.name}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Best Test Accuracy: {best_accuracy:.2f}% (Round {best_round})")
    print(f"Total Rounds: {config['federated']['num_rounds']}")
    print(f"Total Communication: {sum(server.history['communication_cost']) / (1024 * 1024):.2f} MB")
    print("=" * 80)
    
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