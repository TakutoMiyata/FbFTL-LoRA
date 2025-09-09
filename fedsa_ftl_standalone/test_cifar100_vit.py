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
    
    def __init__(self, client_id, model, device, privacy_mechanism=None):
        super().__init__(client_id, model, device, privacy_mechanism)
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
            'num_clients': 3,  # Override for testing
            'model_type': 'vit'  # Important: Use ViT-specific transforms
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
        'secure_aggregation': False,
        'use_opacus': True,  # Use Opacus for efficient DP-SGD
        'total_rounds': 5  # Total rounds for privacy budget
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
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0)
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
    
    # Create privacy mechanism if enabled
    privacy_mechanism = None
    if config.get('privacy', {}).get('enable_privacy', False):
        from privacy_utils import DifferentialPrivacy
        privacy_mechanism = DifferentialPrivacy(
            epsilon=config['privacy'].get('epsilon', 10.0),
            delta=config['privacy'].get('delta', 1e-5),
            max_grad_norm=config['privacy'].get('max_grad_norm', 0.5),
            total_rounds=config['privacy'].get('total_rounds', 5),
            use_opacus=config['privacy'].get('use_opacus', True)
        )
    
    for client_id in range(config['federated']['num_clients']):
        client_model = create_model_vit(config['model'])
        client = ViTFedSAFTLClient(client_id, client_model, device, privacy_mechanism)
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
                shuffle=True,
                num_workers=config['data'].get('num_workers', 0)
            )
            
            # Update with global A matrices if available
            global_A_params = server.get_global_A_params()
            if global_A_params:
                clients[client_id].update_model(global_A_params)
            
            # Local training
            print(f"Training ViT client {client_id}...")
            
            # Add detailed training progress with privacy information
            if privacy_mechanism and privacy_mechanism.use_opacus:
                print(f"    Training with differential privacy (Opacus)")
            elif privacy_mechanism:
                print(f"    Training with differential privacy (per-sample clipping)")
            else:
                print(f"    Training without privacy protection")
                
            client_result = clients[client_id].train(client_dataloader, config['training'])
            client_updates.append(client_result)
            
            # Show privacy budget if applicable
            if privacy_mechanism:
                epsilon_spent, delta = privacy_mechanism.get_privacy_spent()
                print(f"    Privacy budget spent: ε={epsilon_spent:.2f}, δ={delta:.2e}")
            
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
        avg_train_accuracy = round_stats.get('train_accuracy', 0)
        communication_cost = round_stats.get('communication_cost_mb', 0)
        
        print(f"\nRound {round_idx + 1} Summary (ViT):")
        print(f"  Average Training Accuracy: {avg_train_accuracy:.2f}%")
        print(f"  Average Test Accuracy: {avg_test_accuracy:.2f}%")
        print(f"  Communication Cost: {communication_cost:.2f} MB")
        
        if avg_test_accuracy > best_accuracy:
            best_accuracy = avg_test_accuracy
            print(f"  ** New best ViT test accuracy! **")
        
        # Show cumulative privacy budget if applicable
        if privacy_mechanism and round_idx == 0:
            epsilon_total, delta_total = privacy_mechanism.get_privacy_spent()
            print(f"  Total Privacy Budget: ε≤{config['privacy']['epsilon']:.1f}, δ={delta_total:.2e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("ViT Test Complete!")
    print("=" * 80)
    print(f"Configuration: CIFAR-100 + {config['model']['model_name']} + LoRA")
    print(f"Privacy: {'Enabled (ε=' + str(config['privacy']['epsilon']) + ')' if config.get('privacy', {}).get('enable_privacy') else 'Disabled'}")
    print(f"Best ViT Test Accuracy: {best_accuracy:.2f}%")
    print(f"Total Rounds: {config['federated']['num_rounds']}")
    print(f"Total Communication: {sum(server.history['communication_cost']) / (1024 * 1024):.2f} MB")
    
    # Show final privacy budget if applicable
    if privacy_mechanism:
        final_epsilon, final_delta = privacy_mechanism.get_privacy_spent()
        print(f"Final Privacy Budget Used: ε={final_epsilon:.2f}, δ={final_delta:.2e}")
    
    print("=" * 80)
    
    return best_accuracy


def show_vit_model_info():
    """Display ViT model size and statistics"""
    print("\n" + "=" * 80)
    print("Vision Transformer Model Information")
    print("=" * 80)
    
    # ViT configurations for different sizes
    vit_configs = {
        'vit_small': {
            'num_classes': 100,
            'model_name': 'vit_small',
            'lora_r': 16,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'freeze_backbone': True
        }
    }
    
    # Create temporary device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Show statistics for each ViT variant
    for model_name, config in vit_configs.items():
        try:
            vit_model = create_model_vit(config)
            vit_client = ViTFedSAFTLClient(0, vit_model, device)
            vit_stats = vit_client.get_model_size()
            
            print(f"\n{model_name.upper()} Model:")
            print(f"  Total parameters: {vit_stats['total_params']:,}")
            print(f"  Trainable parameters: {vit_stats['trainable_params']:,}")
            print(f"  Communication parameters: {vit_stats['communication_params']:,}")
            print(f"  Compression ratio: {vit_stats['compression_ratio']:.2f}x")
        except Exception as e:
            print(f"\n{model_name.upper()} Model: Error - {e}")


if __name__ == "__main__":
    try:
        # First show ViT model information
        show_vit_model_info()
        
        # Run ViT test
        best_accuracy = run_minimal_vit_test()
        
        # Success criteria (adjusted for differential privacy)
        privacy_enabled = config.get('privacy', {}).get('enable_privacy', False)
        if privacy_enabled:
            # Lower threshold when privacy is enabled due to accuracy trade-off
            threshold = 8  # Expect at least 8% with privacy on CIFAR-100
            if best_accuracy > threshold:
                print("\n✅ ViT Test PASSED: Model is learning successfully with differential privacy!")
                print(f"Note: CIFAR-100 with DP is very challenging (100 classes + privacy noise)")
                print(f"Random guessing: 1%, Current: {best_accuracy:.1f}% - shows significant learning despite privacy constraints.")
            else:
                print("\n⚠️ ViT Test WARNING: Accuracy is lower than expected with privacy enabled.")
                print("This might be due to strong privacy constraints or minimal training setup.")
                print("ViT models with DP typically need more rounds and careful hyperparameter tuning.")
        else:
            # Higher threshold when privacy is disabled
            threshold = 15
            if best_accuracy > threshold:
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