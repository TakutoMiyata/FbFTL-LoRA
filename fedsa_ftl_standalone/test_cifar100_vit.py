"""
Quick test script for CIFAR-100 with Vision Transformer (ViT)
Tests the ViT model setup and runs a minimal federated learning experiment
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
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from fedsa_ftl_model_vit import create_model_vit
from fedsa_ftl_client import FedSAFTLClient
from fedsa_ftl_server import FedSAFTLServer
from data_utils import prepare_federated_data, get_client_dataloader


class ViTFedSAFTLClient(FedSAFTLClient):
    """
    Extended FedSA-FTL Client specifically for Vision Transformer models
    Inherits from existing client but uses ViT model creation
    """
    
    def __init__(self, client_id, model, device, privacy_config=None):
        super().__init__(client_id, model, device, privacy_config)
        # All existing functionality is inherited


def run_minimal_vit_test():
    """Run a minimal test with 3 clients and 5 rounds using Vision Transformer"""
    
    print("=" * 80)
    print("FedSA-FTL Quick Test: CIFAR-100 with Vision Transformer (ViT)")
    print("=" * 80)
    
    # Test configuration
    config = {
        'seed': 42,
        'use_gpu': torch.cuda.is_available(),
        'experiment': {
            'name': 'test_cifar100_vit',
            'output_dir': 'experiments/test_cifar100_vit'
        },
        'model': {
            'num_classes': 100,  # CIFAR-100 has 100 classes
            'model_name': 'vit_small',  # Use small ViT for faster training
            'lora_r': 16,  # Same as VGG16 for comparison
            'lora_alpha': 16,  # 1:1 ratio
            'lora_dropout': 0.1,
            'freeze_backbone': True
        },
        'data': {
            'dataset_name': 'cifar100',  # Use CIFAR-100
            'data_dir': './data',
            'batch_size': 32,  # Smaller batch for ViT
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
            'learning_rate': 0.001,  # Lower learning rate for ViT stability
            'weight_decay': 0.0001  # Lower weight decay for ViT
        }
    }
    
    # Enable differential privacy for testing
    config['privacy'] = {
        'enable_privacy': True,
        'epsilon': 10.0,  # Higher epsilon for testing
        'delta': 1e-5,
        'max_grad_norm': 0.5,
        'noise_multiplier': None,
        'secure_aggregation': False
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if config['use_gpu'] else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Prepare data
    print("\nPreparing CIFAR-100 data with non-IID split...")
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
    temp_model = create_model_vit(config['model'])
    temp_client = ViTFedSAFTLClient(0, temp_model, device)
    model_stats = temp_client.get_model_size()
    
    print("\nViT Model Statistics:")
    print(f"  Total parameters: {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    print(f"  LoRA A parameters: {model_stats['lora_A_params']:,}")
    print(f"  LoRA B parameters: {model_stats['lora_B_params']:,}")
    print(f"  Communication parameters per round: {model_stats['communication_params']:,}")
    print(f"  Compression ratio: {model_stats['compression_ratio']:.2f}x")
    
    # Create clients
    print(f"\nCreating {config['federated']['num_clients']} ViT clients...")
    clients = []
    for client_id in range(config['federated']['num_clients']):
        client_model = create_model_vit(config['model'])
        client = ViTFedSAFTLClient(client_id, client_model, device, config.get('privacy'))
        clients.append(client)
    
    # Training loop
    print(f"\nStarting ViT federated training for {config['federated']['num_rounds']} rounds...")
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
            print(f"Training ViT client {client_id}...")
            client_result = clients[client_id].train(client_dataloader, config['training'])
            client_updates.append(client_result)
            
            print(f"  Client {client_id} - Loss: {client_result['loss']:.4f}, "
                  f"Accuracy: {client_result['accuracy']:.2f}%")
        
        # Evaluate on test set
        print("\nEvaluating ViT models on test set...")
        client_test_results = []
        for client_id in selected_clients:
            test_result = clients[client_id].evaluate(test_dataloader)
            client_test_results.append(test_result)
            print(f"  Client {client_id} test accuracy: {test_result['accuracy']:.2f}%")
        
        # Server aggregation
        print("\nServer aggregating ViT LoRA matrices...")
        round_stats = server.federated_round(client_updates, client_test_results)
        
        # Print round summary
        avg_test_accuracy = round_stats.get('test_accuracy', 0)
        print(f"\nRound {round_idx + 1} Summary (ViT):")
        print(f"  Average Training Accuracy: {round_stats['train_accuracy']:.2f}%")
        print(f"  Average Test Accuracy: {avg_test_accuracy:.2f}%")
        print(f"  Communication Cost: {round_stats['communication_cost_mb']:.2f} MB")
        
        if avg_test_accuracy > best_accuracy:
            best_accuracy = avg_test_accuracy
            print(f"  ** New best ViT test accuracy! **")
    
    # Final summary
    print("\n" + "=" * 80)
    print("ViT Test Complete!")
    print("=" * 80)
    print(f"Best ViT Test Accuracy: {best_accuracy:.2f}%")
    print(f"Total Communication: {sum(server.history['communication_cost']) / (1024 * 1024):.2f} MB")
    
    return best_accuracy


def compare_model_sizes():
    """Compare model sizes between VGG16 and ViT"""
    print("\n" + "=" * 80)
    print("Model Size Comparison: VGG16 vs ViT")
    print("=" * 80)
    
    # Import VGG16 model for comparison
    from src.fedsa_ftl_model import create_model
    
    # VGG16 config
    vgg_config = {
        'num_classes': 100,
        'model_name': 'vgg16',
        'lora_r': 16,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'freeze_backbone': True
    }
    
    # ViT config
    vit_config = {
        'num_classes': 100,
        'model_name': 'vit_small',
        'lora_r': 16,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'freeze_backbone': True
    }
    
    # Create models
    vgg_model = create_model(vgg_config)
    vit_model = create_model_vit(vit_config)
    
    # Create temporary clients to get model stats
    device = torch.device('cpu')
    vgg_client = FedSAFTLClient(0, vgg_model, device)
    vit_client = ViTFedSAFTLClient(0, vit_model, device)
    
    vgg_stats = vgg_client.get_model_size()
    vit_stats = vit_client.get_model_size()
    
    print(f"\nVGG16 Model:")
    print(f"  Total parameters: {vgg_stats['total_params']:,}")
    print(f"  Trainable parameters: {vgg_stats['trainable_params']:,}")
    print(f"  Communication parameters: {vgg_stats['communication_params']:,}")
    print(f"  Compression ratio: {vgg_stats['compression_ratio']:.2f}x")
    
    print(f"\nViT Small Model:")
    print(f"  Total parameters: {vit_stats['total_params']:,}")
    print(f"  Trainable parameters: {vit_stats['trainable_params']:,}")
    print(f"  Communication parameters: {vit_stats['communication_params']:,}")
    print(f"  Compression ratio: {vit_stats['compression_ratio']:.2f}x")
    
    # Calculate ratios
    param_ratio = vit_stats['total_params'] / vgg_stats['total_params']
    trainable_ratio = vit_stats['trainable_params'] / vgg_stats['trainable_params']
    comm_ratio = vit_stats['communication_params'] / vgg_stats['communication_params']
    
    print(f"\nComparison (ViT/VGG16 ratios):")
    print(f"  Total parameters: {param_ratio:.2f}x")
    print(f"  Trainable parameters: {trainable_ratio:.2f}x")
    print(f"  Communication parameters: {comm_ratio:.2f}x")


if __name__ == "__main__":
    try:
        # First compare model sizes
        compare_model_sizes()
        
        # Run ViT test
        best_accuracy = run_minimal_vit_test()
        
        # Success criteria (similar to VGG16 test)
        if best_accuracy > 15:  # Expect at least 15% for quick test on 100 classes
            print("\n✅ ViT Test PASSED: Model is learning successfully!")
            print(f"Note: CIFAR-100 is challenging with 100 classes vs 10 in CIFAR-10")
            print(f"Random guessing would be 1%, so {best_accuracy:.1f}% shows significant learning.")
        else:
            print("\n⚠️ ViT Test WARNING: Accuracy is lower than expected.")
            print("This might be due to the minimal training setup (only 5 rounds).")
            print("ViT models typically need more rounds to converge than CNNs.")
        
    except Exception as e:
        print(f"\n❌ ViT Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)