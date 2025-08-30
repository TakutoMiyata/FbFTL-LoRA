#!/usr/bin/env python3
"""
Quick demo script for FedSA-FTL (works without external dependencies for structure verification)
"""

import sys
import os

def check_file_structure():
    """Check if all necessary files are present"""
    print("Checking FedSA-FTL file structure...")
    
    required_files = [
        'src/__init__.py',
        'src/fedsa_ftl_model.py',
        'src/fedsa_ftl_client.py', 
        'src/fedsa_ftl_server.py',
        'src/data_utils.py',
        'src/experiment_controller.py',
        'configs/cifar10_vit_base.json',
        'configs/cifar100_vit_challenging.json',
        'main.py',
        'analyze_results.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print(f"\n🎉 All {len(required_files)} required files are present!")
        return True


def check_code_structure():
    """Check basic code structure without importing dependencies"""
    print("\nChecking code structure...")
    
    try:
        # Check if src directory is importable
        sys.path.append('./src')
        
        # Basic syntax check by reading files
        with open('src/fedsa_ftl_model.py', 'r') as f:
            model_code = f.read()
            if 'class FedSAFTLModel' in model_code and 'class LoRALayer' in model_code:
                print("✓ Model classes found")
            else:
                print("❌ Model classes missing")
                return False
        
        with open('src/fedsa_ftl_client.py', 'r') as f:
            client_code = f.read()
            if 'class FedSAFTLClient' in client_code:
                print("✓ Client class found")
            else:
                print("❌ Client class missing")
                return False
        
        with open('src/fedsa_ftl_server.py', 'r') as f:
            server_code = f.read()
            if 'class FedSAFTLServer' in server_code:
                print("✓ Server class found")
            else:
                print("❌ Server class missing")
                return False
        
        with open('src/data_utils.py', 'r') as f:
            data_code = f.read()
            if 'class DirichletDataSplitter' in data_code:
                print("✓ Data utilities found")
            else:
                print("❌ Data utilities missing")
                return False
        
        with open('src/experiment_controller.py', 'r') as f:
            exp_code = f.read()
            if 'class FedSAFTLExperiment' in exp_code:
                print("✓ Experiment controller found")
            else:
                print("❌ Experiment controller missing")
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ Code structure check failed: {e}")
        return False


def check_config_files():
    """Check configuration files"""
    print("\nChecking configuration files...")
    
    try:
        import json
        
        configs = ['configs/cifar10_vit_base.json', 'configs/cifar100_vit_challenging.json']
        
        for config_path in configs:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                required_keys = ['experiment_name', 'dataset', 'model', 'federated', 'training']
                missing_keys = [key for key in required_keys if key not in config]
                
                if missing_keys:
                    print(f"❌ {config_path} missing keys: {missing_keys}")
                    return False
                else:
                    print(f"✓ {config_path} is valid")
        
        return True
    
    except Exception as e:
        print(f"❌ Config check failed: {e}")
        return False


def show_implementation_summary():
    """Show summary of the implementation"""
    print("\n" + "="*60)
    print("FedSA-FTL IMPLEMENTATION SUMMARY")
    print("="*60)
    
    print("""
🏗️  ARCHITECTURE COMPONENTS:
   ├── FedSAFTLModel: Hybrid model with frozen backbone + LoRA head
   ├── LoRALayer: Low-rank adaptation implementation
   ├── FedSAFTLHead: Task-specific head with LoRA adaptation
   └── Model factories: create_vision_model, create_nlp_model

👥 CLIENT-SIDE COMPONENTS:
   ├── FedSAFTLClient: Individual client implementation
   ├── FedSAFTLClientManager: Multi-client management
   └── Local training with selective parameter sharing

🖥️  SERVER-SIDE COMPONENTS:
   ├── FedSAFTLServer: Central server with aggregation
   ├── FedAvg aggregation for LoRA A parameters
   └── Global model evaluation and statistics

📊 DATA & EXPERIMENTS:
   ├── DirichletDataSplitter: Non-IID data distribution
   ├── CIFAR-10/100 federated data loaders
   ├── FedSAFTLExperiment: Complete experiment controller
   └── Result analysis and visualization tools

🔧 KEY FEATURES:
   ✓ 50% communication reduction vs standard LoRA FL
   ✓ Personalization through local LoRA B matrices
   ✓ Frozen backbone for computational efficiency
   ✓ Non-IID robustness through selective aggregation
   ✓ Comprehensive experiment management
   ✓ Detailed analysis and visualization tools
""")
    
    print("="*60)
    print("USAGE:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run experiment: python main.py --config configs/cifar10_vit_base.json") 
    print("  3. Analyze results: python analyze_results.py --results results/experiment.json --mode single")
    print("="*60)


def main():
    """Main verification function"""
    print("FedSA-FTL Implementation Verification")
    print("=" * 50)
    
    checks = [
        check_file_structure,
        check_code_structure,
        check_config_files
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        if check():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Verification Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Implementation verification successful!")
        show_implementation_summary()
        return 0
    else:
        print("❌ Some checks failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
