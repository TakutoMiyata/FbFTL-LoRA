# FedSA-FTL: Federated Share-A Transfer Learning

A state-of-the-art federated learning framework that combines frozen backbone transfer learning with LoRA-based selective parameter aggregation for communication-efficient and privacy-preserving distributed training.

## 🌟 Key Features

- **Dual Model Architecture Support**: VGG16 and Vision Transformer (ViT) backends
- **Transfer Learning with Frozen Backbone**: Leverages pre-trained models with frozen feature extractors for efficient transfer learning
- **LoRA-based Selective Aggregation**: Applies Low-Rank Adaptation (LoRA) to classification heads with A/B matrix separation
- **Extreme Communication Efficiency**: Shares only LoRA A-matrices, achieving >100x reduction in communication overhead
- **GPU Acceleration**: Full CUDA support for accelerated training
- **Differential Privacy (DP-SGD)**: Implements rigorous per-sample gradient clipping with calibrated noise addition
- **Client Personalization**: B-matrices remain local for client-specific adaptation while A-matrices are globally aggregated
- **Non-IID Data Support**: Handles heterogeneous data distributions using Dirichlet allocation
- **Flexible Optimizer Support**: Configurable optimizers (SGD, Adam, AdamW) for different scenarios
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

## 🚀 Quick Start

### Basic Training

```bash
# CIFAR-10 with VGG16 (recommended)
python main.py --config configs/cifar10_vgg16_base.yaml

# CIFAR-100 with VGG16 and differential privacy
python main.py --config configs/cifar100_vgg16_private.yaml

# CIFAR-100 with Vision Transformer (ViT)
python quickstart_vit.py

# Custom GPU and seed
python main.py --config configs/cifar10_vgg16_base.yaml --gpu 0 --seed 42
```

### Test Scripts

```bash
# Quick test with VGG16 on CIFAR-10
python test_cifar10_vgg16.py

# Test CIFAR-100 with VGG16
python test_cifar100_vgg16.py

# Test CIFAR-100 with Vision Transformer
python test_cifar100_vit.py

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
  model_type: "vit"            # Vision Transformer
  img_size: 32                 # Input image size
  patch_size: 4                # Patch size (32/4 = 64 patches)
  embed_dim: 384               # Embedding dimension
  depth: 12                    # Number of transformer blocks
  num_heads: 6                 # Number of attention heads
  mlp_ratio: 4.0              # MLP hidden dim ratio
```

### Training Configuration

```yaml
training:
  local_epochs: 5              # Local training epochs per round
  learning_rate: 0.01-0.1      # Higher for LoRA fine-tuning
  weight_decay: 0.0005         # L2 regularization
  optimizer: "sgd"             # Options: "sgd", "adam", "adamw"
  momentum: 0.9                # For SGD
  betas: [0.9, 0.999]         # For Adam/AdamW
  batch_size: 32              # Local batch size
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

### Privacy Configuration (DP-SGD)

```yaml
privacy:
  enable_privacy: true         # Enable differential privacy
  epsilon: 10.0               # Privacy budget (ε)
  delta: 1e-5                 # Privacy parameter (δ)
  max_grad_norm: 0.5-1.0      # Per-sample gradient clipping bound
  total_rounds: 100           # For budget allocation
```

## 🔒 Differential Privacy Implementation

Our implementation follows rigorous DP-SGD principles:

1. **Per-Sample Gradient Clipping**: Each sample's gradient is individually clipped to bound sensitivity
2. **Gradient Averaging**: Clipped gradients are averaged across samples
3. **Calibrated Noise Addition**: Gaussian noise scaled by `(σ × C) / n` where:
   - σ = noise multiplier
   - C = clipping bound
   - n = number of samples

```python
# Pseudocode for DP-SGD
for sample in batch:
    grad = compute_gradient(sample)
    clipped_grad = clip_to_norm(grad, max_norm=C)
    accumulate(clipped_grad)

avg_grad = sum(clipped_grads) / n_samples
noisy_grad = avg_grad + N(0, (σ×C/n)²)
update_model(noisy_grad)
```

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
│   ├── cifar10_vgg16_base.yaml
│   ├── cifar100_vgg16_private.yaml
│   └── cifar100_vit_base.yaml
├── experiments/                    # Training outputs and checkpoints
├── data/                          # Dataset cache directory
├── main.py                        # Main training script
├── quickstart_vit.py              # ViT quick start script
├── quickstart_vgg16.py            # VGG16 quick start script
└── test_*.py                      # Test scripts for various configurations
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

1. **GPU Memory Issues**:
   - Reduce `batch_size` to 16 or 8
   - Use gradient accumulation if needed
   - Monitor GPU usage with `nvidia-smi`

2. **High Test Loss with DP**: 
   - Reduce `epsilon` gradually (start with 10-20)
   - Lower `max_grad_norm` to 0.5
   - Use AdamW optimizer for better convergence

3. **Slow Convergence**:
   - Increase `learning_rate` (LoRA allows higher LR)
   - Increase `client_fraction` to 0.5
   - Check data distribution with `verbose: true`

4. **ViT-specific Issues**:
   - Ensure patch_size divides image_size evenly
   - Adjust embed_dim and depth for smaller datasets
   - Use lower learning rate (0.001-0.01) for ViT

## 📈 Advanced Features

### GPU Acceleration
The framework automatically detects and utilizes available GPUs:
```python
# Automatic GPU detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify GPU in config
use_gpu: true
gpu_id: 0  # For multi-GPU systems
```

### Optimizer Selection
```yaml
# For stable training (default)
optimizer: "sgd"
momentum: 0.9

# For faster convergence with ViT
optimizer: "adamw"
betas: [0.9, 0.999]
weight_decay: 0.05
```

### Non-IID Data Control
```yaml
data_split: "non_iid"
alpha: 0.1  # Very heterogeneous
# alpha: 0.5  # Moderately heterogeneous  
# alpha: 1.0  # Less heterogeneous
# alpha: 100  # Nearly IID
```

### Model Selection
```python
# VGG16 for CIFAR-10
model = create_model(num_classes=10, model_name='vgg16')

# Vision Transformer for CIFAR-100
model = create_model_vit(num_classes=100, img_size=32, patch_size=4)
```

## 🎯 Key Innovations

1. **Dual Architecture Support**: Both CNN (VGG16) and Transformer (ViT) backends
2. **GPU-Optimized Privacy**: Efficient DP-SGD implementation with CUDA acceleration
3. **Selective Parameter Sharing**: Only A-matrices communicated, drastically reducing bandwidth
4. **Personalization via B-matrices**: Client-specific adaptations preserved locally
5. **Transfer Learning**: Frozen backbones preserve pre-trained knowledge
6. **Flexible Architecture**: Supports various optimizers and privacy configurations

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