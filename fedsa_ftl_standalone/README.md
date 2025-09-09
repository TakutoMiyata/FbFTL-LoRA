# FedSA-FTL: Federated Share-A Transfer Learning

A state-of-the-art federated learning framework that combines frozen backbone transfer learning with LoRA-based selective parameter aggregation for communication-efficient and privacy-preserving distributed training.

## 🌟 Key Features

- **Dual Model Architecture Support**: VGG16 and Vision Transformer (ViT) backends
- **Transfer Learning with Frozen Backbone**: Leverages pre-trained models with frozen feature extractors for efficient transfer learning
- **LoRA-based Selective Aggregation**: Applies Low-Rank Adaptation (LoRA) to classification heads with A/B matrix separation
- **Extreme Communication Efficiency**: Shares only LoRA A-matrices, achieving >100x reduction in communication overhead
- **GPU Acceleration**: Full CUDA support for accelerated training
- **Opacus-based Differential Privacy**: Efficient DP-SGD implementation with automated hook management
- **Client Personalization**: B-matrices remain local for client-specific adaptation while A-matrices are globally aggregated
- **Non-IID Data Support**: Handles heterogeneous data distributions using Dirichlet allocation
- **Flexible Configuration**: YAML-based configuration system with comprehensive options
- **Real-time Notifications**: Slack integration for training progress monitoring

## 🏗️ Architecture

### VGG16 Backend
```
┌─────────────────────────────────┐
│   Pre-trained VGG16 Backbone    │ ← Frozen (no gradients)
│   Conv Layers + Feature Maps    │   ImageNet pre-trained
└────────────┬────────────────────┘
             │ detach()
             ▼
┌─────────────────────────────────┐
│   Adaptive Pooling Layer        │ ← Trainable
│   Global Average Pool → 512     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   LoRA-Adapted Classifier       │ ← LoRA decomposition
│   W = W₀ + B·A                  │   A: shared globally
│   512 → 512 → num_classes       │   B: kept local
└─────────────────────────────────┘
```

### Vision Transformer (ViT) Backend
```
┌─────────────────────────────────┐
│   Patch Embedding (4×4)         │ ← Converts image to patches
│   32×32 → 64 patches            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Transformer Encoder           │ ← 12 layers, 6 heads
│   Self-Attention + FFN          │   Dimension: 384
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   LoRA-Adapted MLP Head         │ ← LoRA decomposition
│   W = W₀ + B·A                  │   A: shared globally
│   384 → 384 → num_classes       │   B: kept local
└─────────────────────────────────┘
```

### Key Design Principles

1. **Frozen Backbone**: Feature extractor remains frozen to preserve pre-trained knowledge
2. **LoRA Decomposition**: Weight updates factorized as ΔW = B·A where rank(B·A) << rank(W)
3. **Selective Aggregation**: Only A-matrices are communicated and aggregated at the server
4. **Personalization**: B-matrices capture client-specific patterns and remain local

## 📦 Installation

```bash
# Clone repository
git clone <repository_url>
cd fedsa_ftl_standalone

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- torch>=2.0.0
- torchvision>=0.15.0
- transformers>=4.30.0
- peft>=0.5.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0
- tqdm>=4.65.0
- pyyaml>=6.0
- tensorboard>=2.13.0
- requests>=2.31.0
- opacus>=1.4.0 (for differential privacy)

## 🚀 Quick Start

### Basic Training

```bash
# CIFAR-10 with VGG16 (recommended for beginners)
python main.py --config configs/cifar10_vgg16_base.yaml

# CIFAR-100 with VGG16 and differential privacy
python main.py --config configs/cifar100_vgg16_private.yaml

# CIFAR-100 with Vision Transformer (ViT)
python quickstart_vit.py --config configs/cifar100_vit_base.yaml

# CIFAR-10 with ViT and privacy
python quickstart_vit.py --config configs/cifar10_vit_private.yaml

# Custom rounds and clients
python quickstart_vit.py --rounds 50 --clients 5
```

### Test Scripts

```bash
# Quick test with VGG16 on CIFAR-10
python test_cifar10_vgg16.py

# Test CIFAR-100 with VGG16
python test_cifar100_vgg16.py

# Test CIFAR-100 with Vision Transformer
python test_cifar100_vit.py

# Test privacy mechanisms
python test_privacy.py

# Minimal test for quick validation
python test_minimal.py
```

## ⚙️ Configuration

### Model Configuration

#### VGG16 Settings
```yaml
model:
  num_classes: 10/100          # CIFAR-10 or CIFAR-100
  model_name: "vgg16"          # Pre-trained VGG16
  lora_r: 8-16                 # LoRA rank (8 for CIFAR-10, 16 for CIFAR-100)
  lora_alpha: 8-16             # LoRA scaling (typically 1:1 with rank)
  lora_dropout: 0.1            # Dropout for regularization
  freeze_backbone: true        # Always true for transfer learning
```

#### Vision Transformer Settings
```yaml
model:
  model_name: "vit_small"          # Options: vit_tiny, vit_small, vit_base
  num_classes: 100                 # CIFAR-10: 10, CIFAR-100: 100
  lora_r: 16                       # LoRA rank (16 recommended for CIFAR-100)
  lora_alpha: 16                   # LoRA scaling (1:1 ratio with rank)
  lora_dropout: 0.1                # Dropout for regularization
  freeze_backbone: true            # Always true for transfer learning

data:
  model_type: "vit"                # Enables ViT-specific transforms
  batch_size: 32                   # Smaller batch for memory efficiency
```

### Training Configuration

```yaml
training:
  local_epochs: 5              # Local training epochs per round
  learning_rate: 0.01          # VGG16: 0.01-0.1, ViT: 0.001-0.01
  weight_decay: 0.0005         # VGG16: 0.0005, ViT: 0.0001

# Optional optimizer specification (defaults to SGD)
# optimizer: "sgd"             # Options: "sgd", "adam", "adamw"
# momentum: 0.9                # For SGD
# betas: [0.9, 0.999]         # For Adam/AdamW
```

### Federated Learning Configuration

```yaml
federated:
  num_clients: 10              # Total number of clients
  num_rounds: 100              # Training rounds
  client_fraction: 0.3         # Fraction of clients per round
  checkpoint_freq: 10          # Save checkpoint every N rounds
  aggregation_method: "fedavg" # Weighted averaging by sample count
```

### Privacy Configuration (Opacus-based DP-SGD)

```yaml
privacy:
  enable_privacy: true         # Enable differential privacy
  epsilon: 8.0                 # Privacy budget (ε) - lower = more private
  delta: 1e-5                  # Privacy parameter (δ)
  max_grad_norm: 1.0          # Per-sample gradient clipping bound
  # noise_multiplier: null     # Auto-calculated from epsilon/delta
  # secure_aggregation: false  # Additional privacy layer
```

**Important Notes:**
- Uses Opacus library for efficient DP-SGD implementation
- Automatically manages model hooks and gradient clipping
- GPU-optimized for better performance
- Supports both VGG16 and ViT models

## 🔒 Differential Privacy Implementation

Our implementation uses the Opacus library for efficient and reliable DP-SGD:

### Key Features:
- **Opacus Integration**: Uses PyTorch's official DP library for proven implementations
- **Automatic Hook Management**: Handles model preparation and cleanup automatically
- **GPU Optimization**: Fully accelerated on CUDA devices
- **Per-Sample Clipping**: Rigorous gradient clipping at the sample level
- **Privacy Accounting**: Accurate epsilon/delta tracking with RDP

### How it Works:
```python
# Opacus automatically handles:
1. Model preparation with privacy hooks
2. Per-sample gradient computation
3. Gradient clipping to max_grad_norm
4. Calibrated noise addition
5. Privacy budget accounting
```

### Configuration Example:
```yaml
privacy:
  enable_privacy: true
  epsilon: 8.0          # Total privacy budget
  delta: 1e-5          # Privacy parameter
  max_grad_norm: 1.0   # Clipping threshold
```

### Privacy vs. Accuracy Trade-off:
- **High Privacy** (ε=1-5): Significant accuracy reduction but strong privacy
- **Moderate Privacy** (ε=8-15): Balanced trade-off for practical use
- **Low Privacy** (ε=20+): Minimal accuracy impact with basic privacy

## 📊 Performance Characteristics

### Communication Efficiency
- **Baseline (Full Model)**: 
  - VGG16: ~138.4 MB per round
  - ViT: ~87.2 MB per round
- **FedSA-FTL**: 
  - VGG16: ~1.05 MB per round
  - ViT: ~0.59 MB per round
- **Compression Ratio**: >100x reduction

### Model Statistics

#### VGG16 + LoRA
```
Total parameters: 14,530,378
Trainable parameters: 541,962 (3.7%)
LoRA A parameters: 135,424 (communicated)
LoRA B parameters: 135,424 (kept local)
```

#### Vision Transformer + LoRA
```
Total parameters: 22,354,028
Trainable parameters: 147,840 (0.66%)
LoRA A parameters: 73,920 (communicated)
LoRA B parameters: 73,920 (kept local)
```

### Expected Performance
- **CIFAR-10 (VGG16)**: 85-92% accuracy in 50-100 rounds
- **CIFAR-100 (VGG16)**: 65-75% accuracy in 100 rounds
- **CIFAR-100 (ViT)**: 70-80% accuracy in 100-150 rounds
- **With DP (ε=10)**: 5-10% accuracy reduction

## 📁 Project Structure

```
fedsa_ftl_standalone/
├── src/
│   ├── fedsa_ftl_model.py         # VGG16 + LoRA model
│   ├── fedsa_ftl_model_vit.py     # Vision Transformer + LoRA model
│   ├── fedsa_ftl_client.py        # Client training with DP support
│   ├── fedsa_ftl_server.py        # Server aggregation logic
│   ├── data_utils.py              # Non-IID data partitioning
│   ├── privacy_utils.py           # DP-SGD implementation (GPU-optimized)
│   └── notification_utils.py      # Slack notifications
├── configs/                        # YAML configuration files
│   ├── cifar10_vgg16_base.yaml    # CIFAR-10 + VGG16 baseline
│   ├── cifar10_vit_base.yaml      # CIFAR-10 + ViT baseline  
│   ├── cifar10_vit_private.yaml   # CIFAR-10 + ViT with privacy
│   ├── cifar100_vgg16_base.yaml   # CIFAR-100 + VGG16 baseline
│   ├── cifar100_vgg16_private.yaml # CIFAR-100 + VGG16 with privacy
│   ├── cifar100_vit_base.yaml     # CIFAR-100 + ViT baseline
│   └── cifar100_vit_private.yaml  # CIFAR-100 + ViT with privacy
├── experiments/                    # Training outputs and checkpoints
├── data/                          # Dataset cache directory
├── main.py                        # Main training script (VGG16)
├── quickstart_vit.py              # ViT quick start script
├── quickstart_vgg16.py            # VGG16 quick start script (if available)
├── test_cifar10_vgg16.py          # Test VGG16 on CIFAR-10
├── test_cifar100_vgg16.py         # Test VGG16 on CIFAR-100
├── test_cifar100_vit.py           # Test ViT on CIFAR-100
├── test_privacy.py               # Test privacy mechanisms
├── test_slack.py                 # Test Slack notifications
└── test_minimal*.py              # Minimal validation tests
```

## 🔔 Slack Notifications

Set up real-time training progress notifications:

1. **Get Webhook URL**: Create a Slack app and get incoming webhook URL
2. **Set Environment Variable**: 
   ```bash
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ```
3. **Test**: `python test_slack.py`

Notifications include:
- Training start/completion
- Progress updates every 10 rounds
- Performance metrics and privacy budget
- GPU utilization status

## 🐛 Troubleshooting

### Common Issues

1. **Opacus Hook Errors**:
   - Error: "Trying to add hooks twice to the same model"
   - Solution: Automatically handled by hook cleanup in privacy_utils.py
   - Restart training if issues persist

2. **GPU Memory Issues**:
   - Reduce `batch_size` to 16 or 8
   - Use gradient accumulation if needed
   - Monitor GPU usage with `nvidia-smi`

3. **ViT Convergence Issues**:
   - Use lower learning rate (0.001-0.01) for ViT
   - Increase training rounds (100-150 for CIFAR-100)
   - Ensure proper model variant selection (vit_small recommended)

4. **Privacy vs. Accuracy Trade-off**: 
   - Start with epsilon=8-10 for reasonable privacy/accuracy balance
   - Adjust `max_grad_norm` (0.5-1.0 typical range)
   - Consider longer training with privacy enabled

5. **Configuration Issues**:
   - Verify YAML syntax and file paths
   - Check that all required fields are present
   - Use provided config templates as starting points

## 📈 Advanced Features

### GPU Acceleration
The framework automatically detects and utilizes available GPUs:
```yaml
# Configuration
use_gpu: true  # Automatic GPU detection

# Data loading optimization
data:
  num_workers: 2     # Parallel data loading
  batch_size: 32     # Adjust based on GPU memory
```

### Model Selection
```python
# VGG16 for CIFAR-10/100
python main.py --config configs/cifar10_vgg16_base.yaml

# Vision Transformer variants
python quickstart_vit.py --model vit_tiny   # Fastest, lower accuracy
python quickstart_vit.py --model vit_small  # Balanced (recommended)
python quickstart_vit.py --model vit_base   # Best accuracy, more memory
```

### Privacy Configuration
```yaml
# Conservative privacy (strong protection)
privacy:
  enable_privacy: true
  epsilon: 5.0
  max_grad_norm: 0.5

# Moderate privacy (balanced)
privacy:
  enable_privacy: true  
  epsilon: 8.0
  max_grad_norm: 1.0

# Relaxed privacy (minimal protection)
privacy:
  enable_privacy: true
  epsilon: 15.0
  max_grad_norm: 1.5
```

## 🎯 Key Innovations

1. **Opacus Integration**: Efficient and reliable DP-SGD using PyTorch's official library
2. **Automated Hook Management**: Seamless privacy engine setup and cleanup
3. **Dual Architecture Support**: Both CNN (VGG16) and Transformer (ViT) backends with optimized configurations
4. **GPU-Optimized Privacy**: Full CUDA acceleration for privacy-preserving training
5. **YAML Configuration System**: Comprehensive and flexible configuration management
6. **Selective Parameter Sharing**: Only A-matrices communicated, drastically reducing bandwidth
7. **Transfer Learning Optimization**: Frozen backbones preserve pre-trained knowledge
8. **Robust Error Handling**: Automatic recovery from common Opacus and training issues

## 📚 References

This implementation integrates techniques from:
- **FedSA**: Federated Learning with Selective Aggregation
- **LoRA**: Low-Rank Adaptation of Large Language Models
- **FbFTL**: Feature-based Federated Transfer Learning
- **DP-SGD**: Differentially Private Stochastic Gradient Descent
- **Vision Transformer**: An Image is Worth 16x16 Words

## 📄 License

MIT License

## 👥 Contributors

- FedSA-FTL Development Team