#!/usr/bin/env python3
"""
Simple test script to verify FedSA-FTL implementation
"""

import sys
import os
import torch
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_model_creation():
    """Test basic model creation"""
    print("Testing model creation...")
    
    from fedsa_ftl_model import create_vision_model
    
    try:
        model = create_vision_model(model_name='resnet18', num_classes=10, lora_rank=4)
        print("✓ Vision model created successfully")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
        print("✓ Forward pass successful")
        
        # Test LoRA parameter extraction
        lora_A = model.get_lora_A_parameters()
        lora_B = model.get_lora_B_parameters()
        print(f"✓ LoRA A parameters: {len(lora_A)} tensors")
        print(f"✓ LoRA B parameters: {len(lora_B)} tensors")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_data_splitting():
    """Test data splitting functionality"""
    print("\nTesting data splitting...")
    
    from data_utils import DirichletDataSplitter
    from torchvision import datasets, transforms
    
    try:
        # Create a small dummy dataset
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Use a very small subset for testing
        full_dataset = datasets.CIFAR10(root='./test_data', train=True, download=True, transform=transform)
        
        # Take only first 100 samples for testing
        indices = list(range(100))
        from torch.utils.data import Subset
        test_dataset = Subset(full_dataset, indices)
        
        splitter = DirichletDataSplitter(alpha=0.5)
        client_indices = splitter.split_dataset(test_dataset, num_clients=5, num_classes=10)
        
        print(f"✓ Split dataset into {len(client_indices)} clients")
        print(f"✓ Samples per client: {[len(indices) for indices in client_indices]}")
        
        # Test distribution analysis
        distribution = splitter.get_client_class_distribution(test_dataset, client_indices, 10)
        print(f"✓ Distribution analysis completed for {len(distribution)} clients")
        
        return True
    except Exception as e:
        print(f"✗ Data splitting failed: {e}")
        return False


def test_client_server_setup():
    """Test client and server creation"""
    print("\nTesting client-server setup...")
    
    try:
        from fedsa_ftl_model import create_vision_model
        from fedsa_ftl_client import FedSAFTLClient
        from fedsa_ftl_server import FedSAFTLServer
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        dummy_data = torch.randn(20, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (20,))
        dummy_dataset = TensorDataset(dummy_data, dummy_labels)
        dummy_loader = DataLoader(dummy_dataset, batch_size=4)
        
        # Create model
        model = create_vision_model(model_name='resnet18', num_classes=10, lora_rank=4)
        
        # Create client
        client = FedSAFTLClient(
            client_id=0,
            model=model,
            train_loader=dummy_loader,
            device='cpu'
        )
        print("✓ Client created successfully")
        
        # Create server
        server_model = create_vision_model(model_name='resnet18', num_classes=10, lora_rank=4)
        server = FedSAFTLServer(
            model=server_model,
            test_loader=dummy_loader,
            device='cpu'
        )
        print("✓ Server created successfully")
        
        # Test communication
        lora_A = client.get_lora_A_for_aggregation()
        print(f"✓ Client LoRA A extraction: {len(lora_A)} parameters")
        
        global_lora_A = server.get_global_lora_A()
        client.update_global_lora_A(global_lora_A)
        print("✓ LoRA A communication successful")
        
        return True
    except Exception as e:
        print(f"✗ Client-server setup failed: {e}")
        return False


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    try:
        from fedsa_ftl_model import create_vision_model
        from fedsa_ftl_client import FedSAFTLClient
        from torch.utils.data import DataLoader, TensorDataset
        import torch.nn as nn
        
        # Create dummy data
        dummy_data = torch.randn(8, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (8,))
        dummy_dataset = TensorDataset(dummy_data, dummy_labels)
        dummy_loader = DataLoader(dummy_dataset, batch_size=4)
        
        # Create model and client
        model = create_vision_model(model_name='resnet18', num_classes=10, lora_rank=4)
        client = FedSAFTLClient(
            client_id=0,
            model=model,
            train_loader=dummy_loader,
            device='cpu'
        )
        
        # Test training
        criterion = nn.CrossEntropyLoss()
        training_stats = client.local_train(local_epochs=1, criterion=criterion)
        
        print(f"✓ Training completed")
        print(f"  - Loss: {training_stats['loss']:.4f}")
        print(f"  - Accuracy: {training_stats['accuracy']:.2f}%")
        print(f"  - Samples: {training_stats['num_samples']}")
        
        # Test evaluation
        eval_stats = client.evaluate(criterion=criterion)
        print(f"✓ Evaluation completed")
        print(f"  - Val Loss: {eval_stats['val_loss']:.4f}")
        print(f"  - Val Accuracy: {eval_stats['val_accuracy']:.2f}%")
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False


def main():
    """Run all tests"""
    print("FedSA-FTL Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        test_model_creation,
        test_data_splitting,
        test_client_server_setup,
        test_training_step
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FedSA-FTL implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
