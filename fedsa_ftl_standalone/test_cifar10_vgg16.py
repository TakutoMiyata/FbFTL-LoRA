"""
Quick test script for CIFAR-10 with VGG16
Tests the model setup and runs a minimal federated learning experiment
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.fedsa_ftl_model import create_model
from src.fedsa_ftl_client import FedSAFTLClient
from src.fedsa_ftl_server import FedSAFTLServer
from src.data_utils import prepare_federated_data, get_client_dataloader


def run_minimal_test():
    """Run a minimal test with 3 clients and 5 rounds"""
    
    print("=" * 80)
    print("FedSA-FTL Quick Test: CIFAR-10 with VGG16")
    print("=" * 80)
    
    # Test configuration
    config = {
        'seed': 42,
        'use_gpu': torch.cuda.is_available(),
        'experiment': {
            'name': 'test_cifar10_vgg16',
            'output_dir': 'experiments/test_cifar10_vgg16'
        },
        'model': {
            'num_classes': 10,
            'model_name': 'vgg16',
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'freeze_backbone': True
        },
        'data': {
            'dataset_name': 'cifar10',
            'data_dir': './data',
            'batch_size': 32,
            'num_workers': 2,
            'data_split': 'non_iid',
            'alpha': 0.5,
            'verbose': True,
            'num_clients': 3  # Override for testing
        },
        'federated': {
            'num_clients': 3,
            'num_rounds': 5,
            'client_fraction': 1.0,  # Use all clients for testing
            'aggregation_method': 'fedavg'
        },
        'training': {
            'local_epochs': 2,  # Fewer epochs for testing
            'learning_rate': 0.01,
            'weight_decay': 0.0001
        }
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if config['use_gpu'] else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Prepare data
    print("\nPreparing CIFAR-10 data with non-IID split...")
    trainset, testset, client_indices = prepare_federated_data(config['data'])
    
    # Create test dataloader
    test_dataloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False
    )
    
    # Initialize server
    print("\nInitializing server...")
    server = FedSAFTLServer(device)
    
    # Get model statistics
    temp_model = create_model(config['model'])
    temp_client = FedSAFTLClient(0, temp_model, device)
    model_stats = temp_client.get_model_size()
    
    print("\nModel Statistics:")
    print(f"  Total parameters: {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    print(f"  LoRA A parameters: {model_stats['lora_A_params']:,}")
    print(f"  LoRA B parameters: {model_stats['lora_B_params']:,}")
    print(f"  Communication parameters per round: {model_stats['communication_params']:,}")
    print(f"  Compression ratio: {model_stats['compression_ratio']:.2f}x")
    
    # Create clients
    print(f"\nCreating {config['federated']['num_clients']} clients...")
    clients = []
    for client_id in range(config['federated']['num_clients']):
        client_model = create_model(config['model'])
        client = FedSAFTLClient(client_id, client_model, device)
        clients.append(client)
    
    # Training loop
    print(f"\nStarting federated training for {config['federated']['num_rounds']} rounds...")
    print("=" * 80)
    
    best_accuracy = 0
    
    for round_idx in range(config['federated']['num_rounds']):
        print(f"\n[Round {round_idx + 1}/{config['federated']['num_rounds']}]")
        
        # All clients participate in test
        selected_clients = list(range(config['federated']['num_clients']))
        
        # Client updates
        client_updates = []
        
        for client_id in selected_clients:
            # Get client's data
            client_dataloader = get_client_dataloader(
                trainset, 
                client_indices[client_id],
                config['data']['batch_size'],
                shuffle=True
            )
            
            # Update with global A matrices if available
            global_A_params = server.get_global_A_params()
            if global_A_params:
                clients[client_id].update_model(global_A_params)
            
            # Local training
            print(f"Training client {client_id}...")
            client_result = clients[client_id].train(client_dataloader, config['training'])
            client_updates.append(client_result)
            
            print(f"  Client {client_id} - Loss: {client_result['loss']:.4f}, "
                  f"Accuracy: {client_result['accuracy']:.2f}%")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        client_test_results = []
        for client_id in selected_clients:
            test_result = clients[client_id].evaluate(test_dataloader)
            client_test_results.append(test_result)
            print(f"  Client {client_id} test accuracy: {test_result['accuracy']:.2f}%")
        
        # Server aggregation
        print("\nServer aggregating A matrices...")
        round_stats = server.federated_round(client_updates, client_test_results)
        
        # Print round summary
        avg_test_accuracy = round_stats.get('test_accuracy', 0)
        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Average Training Accuracy: {round_stats['train_accuracy']:.2f}%")
        print(f"  Average Test Accuracy: {avg_test_accuracy:.2f}%")
        print(f"  Communication Cost: {round_stats['communication_cost_mb']:.2f} MB")
        
        if avg_test_accuracy > best_accuracy:
            best_accuracy = avg_test_accuracy
            print(f"  ** New best test accuracy! **")
    
    # Final summary
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Total Communication: {sum(server.history['communication_cost']) / (1024 * 1024):.2f} MB")
    
    return best_accuracy


if __name__ == "__main__":
    try:
        best_accuracy = run_minimal_test()
        
        # Success criteria
        if best_accuracy > 30:  # Expect at least 30% for quick test
            print("\n✅ Test PASSED: Model is learning successfully!")
        else:
            print("\n⚠️ Test WARNING: Accuracy is lower than expected.")
            print("This might be due to the minimal training setup (only 5 rounds).")
        
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)