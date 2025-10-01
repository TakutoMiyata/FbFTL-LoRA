#!/usr/bin/env python3
"""
Debug model creation
"""

import yaml
import sys
import os

# Add src to path
sys.path.append('src')

def debug_model_creation():
    config_path = "configs/experiment_configs_non_iid/non-IID-FedSA-LoRA.yaml"
    
    print("=" * 50)
    print("Debug Model Creation")
    print("=" * 50)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded from: {config_path}")
    print(f"Model name: {config['model']['model_name']}")
    print(f"ImageNet style: {config['data'].get('imagenet_style', False)}")
    print(f"Input size: {config['data'].get('input_size', 32)}")
    
    # Check condition
    use_imagenet_style = config['data'].get('imagenet_style', False)
    print(f"use_imagenet_style = {use_imagenet_style}")
    
    if use_imagenet_style:
        print("✅ Should use ImageNet-style model (make_model_with_lora)")
        
        # Try to import and create model
        try:
            from backbones_imagenet import make_model_with_lora
            print("✅ Successfully imported make_model_with_lora")
            
            # Test model creation
            model = make_model_with_lora(config)
            print(f"✅ Model created successfully: {type(model)}")
            
        except Exception as e:
            print(f"❌ Error creating ImageNet-style model: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("❌ Will use CIFAR-style model (create_cifar_resnet_lora)")
        print("This is probably the issue - MobileNetV2 should use ImageNet-style!")

if __name__ == "__main__":
    debug_model_creation()