#!/bin/bash
# Script to setup Python 3.10 environment for TensorFlow Federated compatibility

echo "🔧 Creating Python 3.10 environment for TensorFlow Federated..."
conda create -n FedSA-LoRA-DP-py310 python=3.10 -y

echo "📦 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate FedSA-LoRA-DP-py310

echo "📥 Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Important notes:"
echo "  - This environment uses Python 3.10 (required for TensorFlow Federated)"
echo "  - transformers==4.30.0 (fixed for PyTorch 2.1 compatibility)"
echo "  - opacus==1.4.0 (fixed for PyTorch 2.0-2.1 compatibility)"
echo ""
echo "To use this environment, run:"
echo "  conda activate FedSA-LoRA-DP-py310"
echo ""
echo "Then you can run:"
echo "  python quickstart_bit_tff.py --config configs/experiment_configs_non_iid/bit_tff_cifar100.yaml"
echo "  python quickstart_resnet.py --config configs/experiment_configs_non_iid/non-IID-FedSA-LoRA.yaml"
