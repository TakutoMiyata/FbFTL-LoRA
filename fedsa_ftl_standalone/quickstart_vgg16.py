"""
Quick start script for FedSA-FTL with VGG-16 and CIFAR-100
Runs a small-scale experiment for testing
"""

import torch
import yaml
import os
from pathlib import Path
from datetime import datetime

# Create a minimal configuration for quick testing with VGG-16 and CIFAR-100
quick_config = {
    'seed': 42,
    'use_gpu': torch.cuda.is_available(),
    'experiment': {
        'name': 'quickstart_vgg16_cifar100',
        'description': 'Quick test of FedSA-FTL with VGG-16 on CIFAR-100',
        'output_dir': 'experiments/quickstart_vgg16'
    },
    'model': {
        'num_classes': 100,  # CIFAR-100
        'model_name': 'vgg16',  # VGG-16 backbone
        'lora_r': 4,  # Small rank for quick testing
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'freeze_backbone': True
    },
    'data': {
        'dataset_name': 'cifar100',  # CIFAR-100 dataset
        'data_dir': './data',
        'batch_size': 16,  # Small batch size for quick testing
        'num_workers': 2,
        'data_split': 'non_iid',
        'alpha': 0.5,
        'verbose': True
    },
    'privacy': {
        'enable_privacy': False,  # Disable privacy for quick testing
    },
    'federated': {
        'num_clients': 5,  # Small number for quick testing
        'num_rounds': 5,  # Few rounds for quick testing
        'client_fraction': 0.4,  # Select 40% of clients per round
        'checkpoint_freq': 5,
        'save_best_model': True,
        'aggregation_method': 'fedavg'
    },
    'training': {
        'local_epochs': 1,  # Just 1 epoch for quick testing
        'learning_rate': 0.001,  # 1e-3
        'weight_decay': 0.0001  # 1e-4
    }
}

def run_quickstart():
    """Run a quick test of FedSA-FTL with VGG-16"""
    print("=" * 80)
    print("FedSA-FTL Quick Start with VGG-16 and CIFAR-100")
    print("=" * 80)
    print("\nThis will run a small-scale experiment to test the implementation.")
    print("Configuration:")
    print(f"  - Model: VGG-16 (ImageNet pre-trained)")
    print(f"  - Dataset: CIFAR-100 (100 classes)")
    print(f"  - {quick_config['federated']['num_clients']} clients (selecting {int(quick_config['federated']['client_fraction'] * quick_config['federated']['num_clients'])} per round)")
    print(f"  - {quick_config['federated']['num_rounds']} federated rounds")
    print(f"  - {quick_config['training']['local_epochs']} local epoch(s) per round")
    print(f"  - LoRA rank: {quick_config['model']['lora_r']}")
    print(f"  - GPU: {quick_config['use_gpu']}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    quick_config['experiment']['output_dir'] = f"experiments/quickstart_vgg16/{timestamp}"
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
    print(f"  Best Test Accuracy: {summary.get('best_test_accuracy', 0):.2f}%")
    print(f"  Final Average Accuracy: {summary.get('final_avg_accuracy', 0):.2f}%")
    print(f"  Total Communication: {summary['total_communication_mb']:.2f} MB")
    print(f"  Results saved to: {quick_config['experiment']['output_dir']}")
    
    return server, clients


if __name__ == "__main__":
    try:
        server, clients = run_quickstart()
    except Exception as e:
        print(f"\nError during quickstart: {e}")
        print("\nPlease check that:")
        print("  1. Conda environment is activated: conda activate FbFTL-LoRA")
        print("  2. All dependencies are installed: pip install -r requirements.txt")
        raise