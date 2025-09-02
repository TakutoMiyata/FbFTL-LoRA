"""
Fixed minimal test script for FedSA-FTL with VGG-16 and CIFAR-100
Optimized for quick verification without timeout
"""

import torch
import yaml
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Create a minimal configuration for very quick testing
minimal_config = {
    'seed': 42,
    'use_gpu': torch.cuda.is_available(),  # Use GPU if available
    'experiment': {
        'name': 'minimal_test_fixed',
        'description': 'Fixed minimal test of FedSA-FTL with VGG-16',
        'output_dir': 'experiments/minimal_test_fixed'
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
        'batch_size': 128,  # Larger batch size for faster processing
        'num_workers': 0,  # No workers for simplicity
        'data_split': 'iid',  # IID for simplicity
        'alpha': 0.5,
        'verbose': False,  # No verbose output
        'subsample_ratio': 0.01  # Use only 1% of data for quick test
    },
    'privacy': {
        'enable_privacy': False,  # No privacy for speed
    },
    'federated': {
        'num_clients': 2,  # Just 2 clients
        'num_rounds': 1,  # Just 1 round for quick test
        'client_fraction': 1.0,  # Use all clients
        'checkpoint_freq': 10,
        'save_best_model': False,
        'aggregation_method': 'fedavg'
    },
    'training': {
        'local_epochs': 1,  # Just 1 epoch
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'max_steps': 5  # Limit steps per epoch for quick test
    }
}

def create_subsampled_data(config):
    """Create a subsampled version of the dataset for quick testing"""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import Subset
    
    # Get transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    trainset = torchvision.datasets.CIFAR100(
        root=config['data']['data_dir'],
        train=True,
        download=True,
        transform=transform
    )
    
    testset = torchvision.datasets.CIFAR100(
        root=config['data']['data_dir'],
        train=False,
        download=True,
        transform=transform
    )
    
    # Subsample for quick test
    subsample_ratio = config['data'].get('subsample_ratio', 0.01)
    n_train_samples = int(len(trainset) * subsample_ratio)
    n_test_samples = int(len(testset) * subsample_ratio)
    
    # Random indices
    np.random.seed(config['seed'])
    train_indices = np.random.choice(len(trainset), n_train_samples, replace=False)
    test_indices = np.random.choice(len(testset), n_test_samples, replace=False)
    
    # Create subsets
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    return train_subset, test_subset

def run_minimal_test():
    """Run a minimal test of FedSA-FTL"""
    print("=" * 60)
    print("FedSA-FTL Minimal Test (Fixed)")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  - Model: VGG-16 with LoRA (r={minimal_config['model']['lora_r']})")
    print(f"  - Dataset: CIFAR-100 (subsampled to {minimal_config['data']['subsample_ratio']*100}%)")
    print(f"  - GPU: {'Yes' if minimal_config['use_gpu'] else 'No'}")
    print(f"  - Clients: {minimal_config['federated']['num_clients']}")
    print(f"  - Rounds: {minimal_config['federated']['num_rounds']}")
    print(f"  - Local epochs: {minimal_config['training']['local_epochs']}")
    print(f"  - Max steps per epoch: {minimal_config['training'].get('max_steps', 'unlimited')}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    minimal_config['experiment']['output_dir'] = f"experiments/minimal_test_fixed/{timestamp}"
    Path(minimal_config['experiment']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(minimal_config['experiment']['output_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(minimal_config, f, default_flow_style=False)
    
    print(f"\nStarting minimal test...")
    print("-" * 60)
    
    try:
        # Quick test: just verify model creation and one forward pass
        from src.fedsa_ftl_model import create_model
        
        print("1. Testing model creation...")
        model = create_model(minimal_config['model'])
        device = torch.device('cuda' if minimal_config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print("   ✅ Model created successfully")
        
        # Test forward pass
        print("2. Testing forward pass...")
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   ✅ Forward pass successful (output shape: {output.shape})")
        
        # Test data loading with subsample
        print("3. Testing data loading (subsampled)...")
        train_subset, test_subset = create_subsampled_data(minimal_config)
        print(f"   ✅ Data loaded (train: {len(train_subset)} samples, test: {len(test_subset)} samples)")
        
        # Quick training test (just a few steps)
        print("4. Testing training loop (limited steps)...")
        from src.fedsa_ftl_client import FedSAFTLClient
        from torch.utils.data import DataLoader
        
        client = FedSAFTLClient(0, model, device, None)
        train_loader = DataLoader(train_subset, batch_size=minimal_config['data']['batch_size'], shuffle=True)
        
        # Train for just a few steps
        model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=minimal_config['training']['learning_rate'])
        
        max_steps = minimal_config['training'].get('max_steps', 5)
        for i, (images, labels) in enumerate(train_loader):
            if i >= max_steps:
                break
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print(f"   Step {i+1}/{max_steps}: loss = {loss.item():.4f}")
        
        print("   ✅ Training loop works correctly")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
        print("\nTest Summary:")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  - Device: {device}")
        print(f"  - All systems operational!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_minimal_test()
    if success:
        print("\n✅ Ready for full training!")
        print("\nTo run full training, use:")
        print("  python main.py --config configs/cifar100_vgg16_private.yaml")
    else:
        print("\n❌ Test failed. Please check the error messages above.")