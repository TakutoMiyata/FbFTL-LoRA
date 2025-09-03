# FedSA-FTL: Federated Share-A Transfer Learning

A state-of-the-art federated learning framework that combines frozen backbone transfer learning with LoRA-based selective parameter aggregation for communication-efficient and privacy-preserving distributed training.

## 🌟 Key Features

- **Transfer Learning with Frozen Backbone**: Leverages pre-trained VGG16 models with frozen feature extractors for efficient transfer learning
- **LoRA-based Selective Aggregation**: Applies Low-Rank Adaptation (LoRA) to classification heads with A/B matrix separation
- **Extreme Communication Efficiency**: Shares only LoRA A-matrices, achieving >100x reduction in communication overhead
- **Differential Privacy (DP-SGD)**: Implements rigorous per-sample gradient clipping with calibrated noise addition
- **Client Personalization**: B-matrices remain local for client-specific adaptation while A-matrices are globally aggregated
- **Non-IID Data Support**: Handles heterogeneous data distributions using Dirichlet allocation
- **Flexible Optimizer Support**: Configurable optimizers (SGD, Adam, AdamW) for different scenarios
- **Real-time Notifications**: Slack integration for training progress monitoring

## 🏗️ Architecture

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
│   W = W₀ + B·A                  │   A: shared globally (communication)
│   512 → 512 → num_classes       │   B: kept local (personalization)
└─────────────────────────────────┘
```

### Key Design Principles

1. **Frozen Backbone**: VGG16 feature extractor remains frozen to preserve pre-trained knowledge
2. **LoRA Decomposition**: Weight updates factorized as ΔW = B·A where rank(B·A) << rank(W)
3. **Selective Aggregation**: Only A-matrices are communicated and aggregated at the server
4. **Personalization**: B-matrices capture client-specific patterns and remain local

## 📦 Installation

```bash
# Clone repository
git clone <repository_url>
cd fedsa_ftl_standalone

# Install dependencies
pip install torch torchvision numpy tqdm pyyaml requests

# Optional: For notifications
pip install python-dotenv
```

## 🚀 Quick Start

### Basic Training

```bash
# CIFAR-10 with VGG16 (recommended)
python main.py --config configs/cifar10_vgg16_base.yaml

# CIFAR-100 with differential privacy
python main.py --config configs/cifar100_vgg16_private.yaml

# Custom GPU and seed
python main.py --config configs/cifar10_vgg16_base.yaml --gpu 0 --seed 42
```

### Test Scripts

```bash
# Quick test with 3 clients
python test_cifar10_vgg16.py

# Test CIFAR-100
python test_cifar100_vgg16.py
```

## ⚙️ Configuration

### Model Configuration

```yaml
model:
  num_classes: 10/100          # CIFAR-10 or CIFAR-100
  model_name: "vgg16"          # Pre-trained VGG16
  lora_r: 8-16                 # LoRA rank (8 for CIFAR-10, 16 for CIFAR-100)
  lora_alpha: 8-16             # LoRA scaling (typically 1:1 with rank)
  lora_dropout: 0.1            # Dropout for regularization
  freeze_backbone: true        # Always true for transfer learning
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
- **Baseline (Full Model)**: ~138.4 MB per round (VGG16)
- **FedSA-FTL**: ~1.05 MB per round (A-matrices only)
- **Compression Ratio**: >100x reduction

### Model Statistics (VGG16 + LoRA)
```
Total parameters: 14,530,378
Trainable parameters: 541,962 (3.7%)
LoRA A parameters: 135,424 (communicated)
LoRA B parameters: 135,424 (kept local)
```

### Expected Performance
- **CIFAR-10**: 85-92% accuracy in 50-100 rounds
- **CIFAR-100**: 65-75% accuracy in 100 rounds
- **With DP (ε=10)**: 5-10% accuracy reduction

## 📁 Project Structure

```
fedsa_ftl_standalone/
├── src/
│   ├── fedsa_ftl_model.py      # VGG16 + LoRA model architecture
│   ├── fedsa_ftl_client.py     # Client training with DP support
│   ├── fedsa_ftl_server.py     # Server aggregation logic
│   ├── data_utils.py           # Non-IID data partitioning
│   ├── privacy_utils.py        # DP-SGD implementation
│   └── notification_utils.py   # Slack notifications
├── configs/                     # YAML configuration files
├── experiments/                 # Training outputs
├── main.py                     # Main training script
└── test_*.py                   # Test scripts
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

## 🐛 Troubleshooting

### Common Issues

1. **High Test Loss with DP**: 
   - Reduce `epsilon` gradually (start with 10-20)
   - Lower `max_grad_norm` to 0.5
   - Use AdamW optimizer for better convergence

2. **Memory Issues**:
   - Reduce `batch_size` to 16 or 8
   - Use gradient accumulation if needed

3. **Slow Convergence**:
   - Increase `learning_rate` (LoRA allows higher LR)
   - Increase `client_fraction` to 0.5
   - Check data distribution with `verbose: true`

## 📈 Advanced Features

### Optimizer Selection
```yaml
# For stable training (default)
optimizer: "sgd"
momentum: 0.9

# For faster convergence with many classes
optimizer: "adamw"
betas: [0.9, 0.999]
```

### Non-IID Data Control
```yaml
data_split: "non_iid"
alpha: 0.1  # Very heterogeneous
# alpha: 0.5  # Moderately heterogeneous  
# alpha: 1.0  # Less heterogeneous
# alpha: 100  # Nearly IID
```

## 🎯 Key Innovations

1. **Selective Parameter Sharing**: Only A-matrices are communicated, drastically reducing bandwidth
2. **Personalization via B-matrices**: Client-specific adaptations preserved locally
3. **Rigorous DP Implementation**: Proper per-sample gradient clipping with calibrated noise
4. **Transfer Learning**: Frozen VGG16 backbone preserves ImageNet knowledge
5. **Flexible Architecture**: Supports various optimizers and privacy configurations

## 📚 References

This implementation integrates techniques from:
- **FedSA**: Federated Learning with Selective Aggregation
- **LoRA**: Low-Rank Adaptation of Large Language Models
- **FbFTL**: Feature-based Federated Transfer Learning
- **DP-SGD**: Differentially Private Stochastic Gradient Descent

## 📄 License

[Specify your license here]

## 👥 Contributors

[List contributors here]