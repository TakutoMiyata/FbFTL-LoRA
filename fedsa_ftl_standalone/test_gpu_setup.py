#!/usr/bin/env python3
"""Test GPU setup for quickstart_bit_tff.py"""

import os
import sys

# Add the same GPU setting as quickstart_bit_tff.py
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

print("=" * 80)
print("GPU Configuration Test")
print("=" * 80)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs visible to PyTorch: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test allocation
    print("\nTesting memory allocation on GPU 0 (which is physical GPU 1)...")
    try:
        test_tensor = torch.zeros(1000, 1000).cuda()
        print(f"✅ Successfully allocated tensor on GPU")
        print(f"Current memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Failed to allocate: {e}")
else:
    print("❌ CUDA is not available!")

print("=" * 80)
