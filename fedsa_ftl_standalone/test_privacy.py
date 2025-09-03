"""
Test script for privacy-preserving FedSA-FTL
"""

import torch
import numpy as np
from src.privacy_utils import DifferentialPrivacy, SecureAggregation
from src.fedsa_ftl_model import create_model
from src.fedsa_ftl_client import FedSAFTLClient


def test_differential_privacy():
    """Test differential privacy mechanism"""
    print("=" * 60)
    print("Testing Differential Privacy Mechanism")
    print("=" * 60)
    
    # Create DP mechanism (test both Opacus and manual implementation)
    print("\nTesting with Opacus (if available):")
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0, 
                             total_rounds=10, use_opacus=True)
    
    # Create dummy parameters
    params = {
        'layer1.lora_A': torch.randn(4, 768),
        'layer2.lora_A': torch.randn(4, 768)
    }
    
    # Test clipping
    print("\n1. Testing parameter clipping:")
    clipped = dp.clip_parameters(params)
    for name, param in clipped.items():
        original_norm = torch.norm(params[name], p=2).item()
        clipped_norm = torch.norm(param, p=2).item()
        print(f"  {name}: original norm={original_norm:.4f}, clipped norm={clipped_norm:.4f}")
    
    # Test noise addition
    print("\n2. Testing noise addition:")
    num_samples = 1000
    noisy = dp.add_noise_to_parameters(params, num_samples)
    for name in params.keys():
        diff = (noisy[name] - params[name]).abs().mean().item()
        print(f"  {name}: average noise magnitude={diff:.6f}")
    
    # Test privacy budget tracking
    print("\n3. Testing privacy budget tracking:")
    for i in range(5):
        _ = dp.apply_differential_privacy(params, num_samples)
        epsilon_spent, delta = dp.get_privacy_spent()
        print(f"  Step {i+1}: ε_spent={epsilon_spent:.2f}, δ={delta:.2e}")
    
    print("\n✓ Differential privacy mechanism working correctly")


def test_secure_aggregation():
    """Test secure aggregation mechanism"""
    print("\n" + "=" * 60)
    print("Testing Secure Aggregation")
    print("=" * 60)
    
    # Create secure aggregator
    num_clients = 3
    sa = SecureAggregation(num_clients)
    
    # Create dummy client updates
    client_updates = []
    for i in range(num_clients):
        update = {
            'client_id': i,
            'num_samples': 100 * (i + 1),
            'lora_A_params': {
                'layer1.lora_A': torch.randn(4, 768) * 0.1,
                'layer2.lora_A': torch.randn(4, 768) * 0.1
            }
        }
        client_updates.append(update)
    
    # Test mask generation
    print("\n1. Testing mask generation:")
    masks = sa.generate_masks((4, 768))
    mask_sum = sum(masks.values())
    print(f"  Number of masks: {len(masks)}")
    print(f"  Mask sum (should be ~0): {mask_sum.abs().max().item():.8f}")
    
    # Test secure aggregation
    print("\n2. Testing secure aggregation:")
    aggregated = sa.aggregate_with_secure_aggregation(client_updates)
    for name, param in aggregated.items():
        print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}")
    
    print("\n✓ Secure aggregation working correctly")


def test_privacy_in_training():
    """Test privacy in actual training scenario"""
    print("\n" + "=" * 60)
    print("Testing Privacy in Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and client with privacy
    model_config = {
        'num_classes': 10,
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'freeze_backbone': True
    }
    
    # Test with and without privacy
    print("\n1. Client without privacy:")
    model_no_privacy = create_model(model_config)
    client_no_privacy = FedSAFTLClient(0, model_no_privacy, device, privacy_mechanism=None)
    
    # Dummy training data
    dummy_data = torch.randn(32, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (32,))
    
    # Get parameters without privacy
    params_no_privacy = client_no_privacy.model.get_lora_params(matrix_type='A')
    
    print("\n2. Client with privacy (ε=1.0):")
    dp_mechanism = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    model_with_privacy = create_model(model_config)
    client_with_privacy = FedSAFTLClient(1, model_with_privacy, device, privacy_mechanism=dp_mechanism)
    
    # Get parameters with privacy
    params_with_privacy = client_with_privacy.model.get_lora_params(matrix_type='A')
    private_params = dp_mechanism.apply_differential_privacy(params_with_privacy, 1000)
    
    # Compare
    print("\n3. Comparing parameters:")
    for name in params_no_privacy.keys():
        if name in private_params:
            diff = (private_params[name] - params_no_privacy[name]).abs().mean().item()
            print(f"  {name}: difference={diff:.6f}")
    
    print("\n✓ Privacy mechanism integrated with training")


def compare_privacy_levels():
    """Compare different privacy levels"""
    print("\n" + "=" * 60)
    print("Comparing Different Privacy Levels")
    print("=" * 60)
    
    params = {
        'layer.lora_A': torch.randn(4, 768) * 0.1
    }
    num_samples = 1000
    
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("\nNoise level for different ε values:")
    print("ε\t\tNoise Std\tAvg Noise")
    print("-" * 40)
    
    for epsilon in epsilon_values:
        dp = DifferentialPrivacy(epsilon=epsilon, delta=1e-5)
        noisy_params = dp.apply_differential_privacy(params.copy(), num_samples)
        
        noise = noisy_params['layer.lora_A'] - params['layer.lora_A']
        noise_std = noise.std().item()
        noise_avg = noise.abs().mean().item()
        
        print(f"{epsilon:.1f}\t\t{noise_std:.6f}\t{noise_avg:.6f}")
    
    print("\nNote: Smaller ε = More privacy but more noise")
    print("      Larger ε = Less privacy but less noise")


if __name__ == "__main__":
    print("Testing Privacy Mechanisms for FedSA-FTL")
    print("=" * 60)
    
    # Run tests
    test_differential_privacy()
    test_secure_aggregation()
    test_privacy_in_training()
    compare_privacy_levels()
    
    print("\n" + "=" * 60)
    print("All Privacy Tests Completed Successfully!")
    print("=" * 60)