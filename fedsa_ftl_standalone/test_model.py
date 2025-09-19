"""
Test script to diagnose model issues
"""

import torch
import torch.nn as nn
from src.fedsa_ftl_model import create_model, LoRALinear
from src.data_utils import load_cifar10_data
from torch.utils.data import DataLoader
import numpy as np

def test_model_components():
    """Test individual model components"""
    print("=" * 60)
    print("Testing Model Components")
    print("=" * 60)
    
    # Create model
    config = {
        'num_classes': 10,
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'freeze_backbone': True
    }
    
    model = create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Check frozen backbone
    print("\n1. Checking frozen backbone:")
    for name, param in model.vit.named_parameters():
        if param.requires_grad:
            print(f"  WARNING: {name} is not frozen!")
    print("  All backbone parameters are frozen ✓")
    
    # Check LoRA parameters
    print("\n2. Checking LoRA parameters:")
    lora_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"  Found LoRA layer: {name}")
            print(f"    - A shape: {module.lora_A.shape}")
            print(f"    - B shape: {module.lora_B.shape}")
            print(f"    - A requires_grad: {module.lora_A.requires_grad}")
            print(f"    - B requires_grad: {module.lora_B.requires_grad}")
            lora_count += 1
    print(f"  Total LoRA layers: {lora_count}")
    
    # Test forward pass
    print("\n3. Testing forward pass:")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test parameter updates
    print("\n4. Testing parameter updates:")
    
    # Get initial A parameters
    initial_A = model.get_lora_params(matrix_type='A')
    print(f"  Initial A parameters keys: {list(initial_A.keys())}")
    
    # Create dummy updated parameters
    updated_A = {}
    for key, value in initial_A.items():
        updated_A[key] = value + 0.1  # Add small change
    
    # Set updated parameters
    model.set_lora_params(updated_A, matrix_type='A')
    
    # Check if parameters were updated
    new_A = model.get_lora_params(matrix_type='A')
    
    all_updated = True
    for key in initial_A.keys():
        diff = (new_A[key] - initial_A[key]).abs().mean().item()
        if diff < 0.09:  # Should be ~0.1
            print(f"  WARNING: {key} was not properly updated! Diff: {diff}")
            all_updated = False
    
    if all_updated:
        print("  All A parameters updated successfully ✓")
    
    # Test actual learning
    print("\n5. Testing actual learning:")
    
    # Load real data
    trainset, testset = load_cifar10_data('./data')
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, pin_memory=True)
    
    # Evaluate before training
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 10:  # Only test first 10 batches for speed
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    initial_acc = 100. * correct / total
    print(f"  Initial accuracy (untrained): {initial_acc:.2f}%")
    
    # Simple training step
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    
    model.train()
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True)
    
    for i, (images, labels) in enumerate(train_loader):
        if i >= 50:  # Train for 50 batches
            break
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"    Batch {i}, Loss: {loss.item():.4f}")
    
    # Evaluate after training
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 10:  # Only test first 10 batches
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    final_acc = 100. * correct / total
    print(f"  Final accuracy (after training): {final_acc:.2f}%")
    print(f"  Improvement: {final_acc - initial_acc:.2f}%")
    
    if final_acc > initial_acc + 5:
        print("  Model is learning properly ✓")
    else:
        print("  WARNING: Model is not learning properly!")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_model_components()