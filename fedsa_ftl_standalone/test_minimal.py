"""
Minimal test script for FedSA-FTL with VGG-16 and CIFAR-100
Very small scale for quick verification
"""

import torch
import yaml
import os
from pathlib import Path
from datetime import datetime

# Create a minimal configuration for very quick testing
minimal_config = {
    'seed': 42,
    'use_gpu': torch.cuda.is_available(),  # Use GPU if available
    'experiment': {
        'name': 'minimal_test',
        'description': 'Minimal test of FedSA-FTL with VGG-16',
        'output_dir': 'experiments/minimal_test'
    },
    'model': {
        'num_classes': 100,  # CIFAR-100
        'model_name': 'vgg16',  # VGG-16 backbone
        'lora_r': 2,  # Very small rank for minimal test
        'lora_alpha': 4,
        'lora_dropout': 0.1,
        'freeze_backbone': True
    },
    'data': {
        'dataset_name': 'cifar100',  # CIFAR-100 dataset
        'data_dir': './data',
        'batch_size': 8,  # Very small batch size
        'num_workers': 0,  # No workers for simplicity
        'data_split': 'iid',  # IID for simplicity
        'alpha': 0.5,
        'verbose': False  # No verbose output
    },
    'privacy': {
        'enable_privacy': False,  # No privacy for speed
    },
    'federated': {
        'num_clients': 2,  # Just 2 clients
        'num_rounds': 2,  # Just 2 rounds
        'client_fraction': 1.0,  # Use all clients
        'checkpoint_freq': 10,
        'save_best_model': False,
        'aggregation_method': 'fedavg'
    },
    'training': {
        'local_epochs': 1,  # Just 1 epoch
        'learning_rate': 0.001,
        'weight_decay': 0.0001
    }
}

def run_minimal_test():
    """Run a minimal test of FedSA-FTL"""
    print("=" * 60)
    print("FedSA-FTL Minimal Test")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  - Model: VGG-16 with LoRA (r={minimal_config['model']['lora_r']})")
    print(f"  - Dataset: CIFAR-100")
    print(f"  - Clients: {minimal_config['federated']['num_clients']}")
    print(f"  - Rounds: {minimal_config['federated']['num_rounds']}")
    print(f"  - Local epochs: {minimal_config['training']['local_epochs']}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    minimal_config['experiment']['output_dir'] = f"experiments/minimal_test/{timestamp}"
    Path(minimal_config['experiment']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(minimal_config['experiment']['output_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(minimal_config, f, default_flow_style=False)
    
    print(f"\nStarting minimal test...")
    print("-" * 60)
    
    try:
        # Import and run main
        from main import main
        server, clients = main(minimal_config)
        
        print("\n" + "=" * 60)
        print("✅ Minimal test completed successfully!")
        print("=" * 60)
        
        # Print results
        summary = server.get_summary_stats()
        print("\nTest Results:")
        print(f"  Total rounds completed: {summary['total_rounds']}")
        print(f"  Communication cost: {summary['total_communication_mb']:.2f} MB")
        print(f"  Model works correctly: Yes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_minimal_test()
    if success:
        print("\n✅ All systems operational! Ready for full training.")
        print("\nTo run full training, use:")
        print("  python main.py --config configs/cifar100_vgg16_private.yaml")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
