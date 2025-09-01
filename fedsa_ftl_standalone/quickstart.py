"""
Quick start script for FedSA-FTL
Runs a small-scale experiment for testing
"""

import torch
import yaml
import os
from pathlib import Path
from datetime import datetime

# Create a minimal configuration for quick testing
quick_config = {
    'seed': 42,
    'use_gpu': torch.cuda.is_available(),
    'experiment': {
        'name': 'quickstart',
        'description': 'Quick test of FedSA-FTL implementation',
        'output_dir': 'experiments/quickstart'
    },
    'model': {
        'num_classes': 10,
        'model_name': 'google/vit-base-patch16-224-in21k',
        'lora_r': 4,  # Small rank for quick testing
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'freeze_backbone': True
    },
    'data': {
        'data_dir': './data',
        'batch_size': 16,  # Small batch size for quick testing
        'num_workers': 2,
        'data_split': 'non_iid',
        'alpha': 0.5,
        'verbose': True
    },
    'federated': {
        'num_clients': 5,  # Moderate number for testing
        'num_rounds': 10,  # More rounds to see convergence
        'client_fraction': 0.4,  # Select 40% of clients per round
        'checkpoint_freq': 5,
        'save_best_model': True,
        'aggregation_method': 'fedavg'
    },
    'training': {
        'local_epochs': 2,  # Few epochs for quick testing
        'learning_rate': 0.001,  # 1e-3
        'weight_decay': 0.0001  # 1e-4
    }
}

def run_quickstart():
    """Run a quick test of FedSA-FTL"""
    print("=" * 80)
    print("FedSA-FTL Quick Start")
    print("=" * 80)
    print("\nThis will run a small-scale experiment to test the implementation.")
    print("Configuration:")
    print(f"  - {quick_config['federated']['num_clients']} clients (selecting {int(quick_config['federated']['client_fraction'] * quick_config['federated']['num_clients'])} per round)")
    print(f"  - {quick_config['federated']['num_rounds']} federated rounds")
    print(f"  - {quick_config['training']['local_epochs']} local epochs per round")
    print(f"  - LoRA rank: {quick_config['model']['lora_r']}")
    print(f"  - GPU: {quick_config['use_gpu']}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    quick_config['experiment']['output_dir'] = f"experiments/quickstart/{timestamp}"
    Path(quick_config['experiment']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(quick_config['experiment']['output_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(quick_config, f, default_flow_style=False)
    
    print(f"\nConfiguration saved to: {config_path}")
    print("\nStarting training...")
    print("-" * 80)
    
    # Import and run main
    from main import main
    server, clients = main(quick_config)
    
    print("\n" + "=" * 80)
    print("Quick start completed successfully!")
    print("=" * 80)
    
    # Print final summary
    summary = server.get_summary_stats()
    print("\nResults Summary:")
    print(f"  Final Test Accuracy: {summary['final_test_accuracy']:.2f}%")
    print(f"  Total Communication: {summary['total_communication_mb']:.2f} MB")
    print(f"  Results saved to: {quick_config['experiment']['output_dir']}")
    
    return server, clients


if __name__ == "__main__":
    try:
        server, clients = run_quickstart()
    except Exception as e:
        print(f"\nError during quickstart: {e}")
        print("\nPlease check that all dependencies are installed:")
        print("  pip install -r requirements.txt")
        raise