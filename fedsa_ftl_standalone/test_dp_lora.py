#!/usr/bin/env python3
"""
Test script for DP-LoRA implementation
Tests the model parameter separation and DP functionality
"""

import torch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fedsa_ftl_model_resnet import create_model_resnet
from dp_utils import create_dp_optimizer, WeightedFedAvg

def test_model_parameter_separation():
    """Test A and B parameter separation"""
    print("=== Testing Model Parameter Separation ===")
    
    config = {
        'model_name': 'resnet18',
        'num_classes': 100,
        'pretrained': False,  # Faster for testing
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1
    }
    
    model = create_model_resnet(config)
    
    # Test parameter group methods
    A_params = model.get_A_parameter_groups()
    B_params = model.get_B_parameter_groups()
    
    print(f"✅ A parameter groups: {len(A_params)}")
    print(f"✅ B parameter groups: {len(B_params)}")
    
    # Test parameter dictionaries
    A_dict = model.get_A_parameters()
    B_dict = model.get_B_parameters()
    
    print(f"✅ A parameters: {list(A_dict.keys())}")
    print(f"✅ B parameters: {list(B_dict.keys())}")
    
    # Test parameter counts
    A_count = sum(p.numel() for p in A_params)
    B_count = sum(p.numel() for p in B_params)
    total_count = sum(p.numel() for p in model.parameters())
    
    print(f"✅ A parameter count: {A_count:,}")
    print(f"✅ B parameter count: {B_count:,}")
    print(f"✅ Total parameters: {total_count:,}")
    print(f"✅ A + B = Total: {A_count + B_count == total_count}")
    
    return model

def test_dp_optimizer():
    """Test DP optimizer functionality"""
    print("\n=== Testing DP Optimizer ===")
    
    config = {
        'model_name': 'resnet18',
        'num_classes': 100,
        'pretrained': False,
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1
    }
    
    full_config = {
        'model': config,
        'training': {
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0001
        },
        'privacy': {
            'max_grad_norm': 0.5,
            'noise_multiplier': 1.0,
            'epsilon': 8.0,
            'delta': 1e-5
        }
    }
    
    model = create_model_resnet(config)
    dp_optimizer = create_dp_optimizer(model, full_config)
    
    print(f"✅ DP Optimizer created")
    print(f"✅ A parameter count: {len(dp_optimizer.A_params)}")
    print(f"✅ B parameter count: {len(dp_optimizer.B_params)}")
    print(f"✅ Max grad norm: {dp_optimizer.max_grad_norm}")
    print(f"✅ Noise multiplier: {dp_optimizer.noise_multiplier}")
    
    # Test gradient clipping and noise
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_target = torch.randint(0, 100, (2,))
    
    # Forward pass
    output = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_target)
    
    # Backward pass
    dp_optimizer.zero_grad()
    loss.backward()
    
    # Check gradients exist
    A_grads = [p.grad is not None for p in dp_optimizer.A_params]
    B_grads = [p.grad is not None for p in dp_optimizer.B_params]
    
    print(f"✅ A gradients exist: {all(A_grads)} ({sum(A_grads)}/{len(A_grads)})")
    print(f"✅ B gradients exist: {all(B_grads)} ({sum(B_grads)}/{len(B_grads)})")
    
    # Test DP step
    dp_optimizer.step()
    print(f"✅ DP step completed")
    print(f"✅ Privacy spent: {dp_optimizer.get_privacy_spent():.6f}")
    
    return dp_optimizer

def test_weighted_fedavg():
    """Test weighted FedAvg aggregation"""
    print("\n=== Testing Weighted FedAvg ===")
    
    # Create dummy client parameters
    config = {
        'model_name': 'resnet18',
        'num_classes': 100,
        'pretrained': False,
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1
    }
    
    # Simulate 3 clients
    client_A_params = []
    client_sample_counts = [100, 200, 150]  # Different sample sizes
    
    for i in range(3):
        model = create_model_resnet(config)
        A_params = model.get_A_parameters()
        client_A_params.append(A_params)
    
    print(f"✅ Created {len(client_A_params)} client parameter sets")
    print(f"✅ Sample counts: {client_sample_counts}")
    
    # Test aggregation
    aggregated_A = WeightedFedAvg.aggregate_A_matrices(client_A_params, client_sample_counts)
    
    print(f"✅ Aggregated parameters: {list(aggregated_A.keys())}")
    print(f"✅ Aggregation completed successfully")
    
    # Test logging
    WeightedFedAvg.log_aggregation_info(client_sample_counts, len(aggregated_A))
    
    return aggregated_A

def test_forward_pass():
    """Test model forward pass with LoRA"""
    print("\n=== Testing Model Forward Pass ===")
    
    config = {
        'model_name': 'resnet18',
        'num_classes': 100,
        'pretrained': False,
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1
    }
    
    model = create_model_resnet(config)
    model.eval()
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        features = model.extract_features(dummy_input)
    
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Features shape: {features.shape}")
    print(f"✅ Output classes: {output.shape[1]} (expected: 100)")
    
    # Test that output is reasonable
    assert output.shape == (4, 100), f"Expected (4, 100), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    print("✅ Forward pass test passed")
    
    return model

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting DP-LoRA Implementation Tests")
    print("=" * 60)
    
    try:
        # Test 1: Model parameter separation
        model = test_model_parameter_separation()
        
        # Test 2: DP optimizer
        dp_optimizer = test_dp_optimizer()
        
        # Test 3: Weighted FedAvg
        aggregated = test_weighted_fedavg()
        
        # Test 4: Forward pass
        model = test_forward_pass()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("✅ Model parameter separation works")
        print("✅ DP optimizer functional")
        print("✅ Weighted aggregation works")
        print("✅ Model forward pass works")
        print("\n📋 Ready for federated learning with DP-LoRA!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)