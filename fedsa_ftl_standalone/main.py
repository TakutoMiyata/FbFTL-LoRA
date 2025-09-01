"""
Main training script for FedSA-FTL
"""

import os
import argparse
import yaml
import torch
import numpy as np
import random
from datetime import datetime
import json
from pathlib import Path

from src.fedsa_ftl_model import create_model
from src.fedsa_ftl_client import FedSAFTLClient
from src.fedsa_ftl_server import FedSAFTLServer
from src.data_utils import prepare_federated_data, get_client_dataloader
from torch.utils.data import DataLoader


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    """Main training function"""
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("\nPreparing federated data...")
    trainset, testset, client_indices = prepare_federated_data(config['data'])
    
    # Create test dataloader
    test_dataloader = DataLoader(
        testset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False,
        num_workers=config['data'].get('num_workers', 2)
    )
    
    # Initialize server (no model needed, only manages A matrices)
    print("\nInitializing server...")
    server = FedSAFTLServer(device)
    
    # Get model size statistics from a temporary model
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
    
    best_test_accuracy = 0
    
    for round_idx in range(config['federated']['num_rounds']):
        print(f"\n[Round {round_idx + 1}/{config['federated']['num_rounds']}]")
        
        # Sample clients for this round
        num_selected = max(1, int(config['federated']['client_fraction'] * config['federated']['num_clients']))
        selected_clients = np.random.choice(range(config['federated']['num_clients']), 
                                           num_selected, replace=False)
        
        print(f"Selected {num_selected} clients: {selected_clients.tolist()}")
        
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
            
            # Update client model with global A matrices
            # Always update, even in first round (to ensure consistency)
            global_A_params = server.get_global_A_params()
            if global_A_params:  # Only update if server has A params
                clients[client_id].update_model(global_A_params)
            
            # Local training
            print(f"\nTraining client {client_id}...")
            client_result = clients[client_id].train(client_dataloader, config['training'])
            client_updates.append(client_result)
            
            print(f"  Client {client_id} - Loss: {client_result['loss']:.4f}, "
                  f"Accuracy: {client_result['accuracy']:.2f}%")
        
        # Evaluate selected clients on test data (with their personalized B matrices)
        print("\nEvaluating clients on test set...")
        client_test_results = []
        for client_id in selected_clients:
            test_result = clients[client_id].evaluate(test_dataloader)
            client_test_results.append(test_result)
            print(f"  Client {client_id} test accuracy: {test_result['accuracy']:.2f}%")
        
        # Server aggregation
        print("\nServer aggregating A matrices...")
        round_stats = server.federated_round(client_updates, client_test_results)
        
        # Print round statistics
        print(f"\nRound {round_idx + 1} Summary:")
        print(f"  Average Training Loss: {round_stats['train_loss']:.4f}")
        print(f"  Average Training Accuracy: {round_stats['train_accuracy']:.2f}%")
        if 'test_loss' in round_stats:
            print(f"  Average Test Loss: {round_stats['test_loss']:.4f}")
            print(f"  Average Test Accuracy: {round_stats['test_accuracy']:.2f}%")
            
            # Track best accuracy
            if round_stats['test_accuracy'] > best_test_accuracy:
                best_test_accuracy = round_stats['test_accuracy']
                print(f"  ** New best test accuracy! **")
                
                # Save best checkpoint
                if config['federated'].get('save_best_model', True):
                    checkpoint_path = os.path.join(
                        config['experiment']['output_dir'],
                        'best_checkpoint.pt'
                    )
                    server.save_checkpoint(checkpoint_path)
        
        print(f"  Communication Cost: {round_stats['communication_cost_mb']:.2f} MB")
        
        # Save checkpoint periodically
        if (round_idx + 1) % config['federated'].get('checkpoint_freq', 10) == 0:
            checkpoint_path = os.path.join(
                config['experiment']['output_dir'],
                f'checkpoint_round_{round_idx + 1}.pt'
            )
            server.save_checkpoint(checkpoint_path)
    
    # Final evaluation with all clients
    print("\n" + "=" * 80)
    print("Final Evaluation with All Clients")
    print("=" * 80)
    
    all_test_results = []
    for client_id in range(config['federated']['num_clients']):
        # Update with final global A params
        global_A_params = server.get_global_A_params()
        if global_A_params:
            clients[client_id].update_model(global_A_params)
        
        # Evaluate
        test_result = clients[client_id].evaluate(test_dataloader)
        all_test_results.append(test_result)
        print(f"Client {client_id} final test accuracy: {test_result['accuracy']:.2f}%")
    
    # Calculate average final performance
    avg_final_accuracy = np.mean([r['accuracy'] for r in all_test_results])
    std_final_accuracy = np.std([r['accuracy'] for r in all_test_results])
    
    print(f"\nAverage final test accuracy: {avg_final_accuracy:.2f}% ± {std_final_accuracy:.2f}%")
    
    # Get final summary
    summary = server.get_summary_stats()
    summary['final_avg_accuracy'] = avg_final_accuracy
    summary['final_std_accuracy'] = std_final_accuracy
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nFinal Summary:")
    print(f"  Total Rounds: {summary['total_rounds']}")
    print(f"  Best Average Test Accuracy: {summary['best_test_accuracy']:.2f}%")
    print(f"  Final Average Test Accuracy: {avg_final_accuracy:.2f}% ± {std_final_accuracy:.2f}%")
    print(f"  Total Communication: {summary['total_communication_mb']:.2f} MB")
    print(f"  Average Communication per Round: {summary['avg_communication_per_round_mb']:.2f} MB")
    
    # Save results
    results = {
        'config': config,
        'model_stats': model_stats,
        'summary': summary,
        'history': server.history,
        'final_client_results': all_test_results
    }
    
    results_path = os.path.join(config['experiment']['output_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return server, clients


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedSA-FTL Training')
    parser.add_argument('--config', type=str, default='configs/cifar10_vit_base.yaml',
                        help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.seed is not None:
        config['seed'] = args.seed
    if args.gpu is not None:
        config['use_gpu'] = True
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Create output directory
    config['experiment']['output_dir'] = os.path.join(
        'experiments',
        config['experiment']['name'],
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    Path(config['experiment']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(config['experiment']['output_dir'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    server, clients = main(config)