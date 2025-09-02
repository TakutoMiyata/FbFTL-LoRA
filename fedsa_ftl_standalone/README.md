# FedSA-FTL: Federated Share-A Transfer Learning

Federated learning framework combining frozen backbone transfer learning with LoRA-based selective parameter aggregation for communication-efficient and privacy-preserving distributed training.

## Features

- **Transfer Learning with Frozen Backbone**: Leverages pre-trained models (ViT, VGG16) with frozen feature extractors
- **LoRA-based Adaptation**: Applies Low-Rank Adaptation only to classification heads
- **Communication Efficiency**: Shares only LoRA A-matrices (>100x reduction in communication)
- **Privacy Protection**: Built-in differential privacy and secure aggregation support
- **Personalization**: B-matrices remain local for client-specific adaptation
- **Non-IID Support**: Handles heterogeneous data distributions with Dirichlet allocation
- **Flexible Models**: Support for Vision Transformer (ViT) and VGG16 architectures
- **Notification System**: Slack/Discord integration for training completion alerts

## Architecture

```
┌─────────────────────────┐
│   Pre-trained Model     │ ← Frozen (no gradients)
│   (ViT/VGG16)          │
│   Feature Extractor     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Classification Head    │ ← LoRA-adapted
│   W = W₀ + B·A          │   A: shared globally
│                         │   B: kept local
└─────────────────────────┘
```

## Installation

```bash
# Clone repository
cd fedsa_ftl_standalone

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Test Implementation
```bash
# Run minimal test with 3 clients
python quickstart.py

# Test with VGG16 model
python quickstart_vgg16.py
```

### Production Training

```bash
# Basic training with default configuration
python main.py --config configs/cifar10_vit_base.yaml

# With custom seed and GPU selection
python main.py --config configs/cifar10_vit_base.yaml --seed 123 --gpu 0

# Challenging non-IID scenario
python main.py --config configs/cifar10_vit_challenging.yaml

# With differential privacy
python main.py --config configs/cifar10_vit_private.yaml

# CIFAR-100 with VGG16
python main.py --config configs/cifar100_vgg16_private.yaml
```

## Configuration Options

### Available Configurations

- `cifar10_vit_base.yaml` - Standard CIFAR-10 training with ViT
- `cifar10_vit_challenging.yaml` - More heterogeneous data distribution
- `cifar10_vit_private.yaml` - With differential privacy enabled
- `cifar100_vgg16_private.yaml` - CIFAR-100 dataset with VGG16 model

### Key Parameters

```yaml
# Model settings
model:
  num_classes: 10              # Number of output classes
  model_name: "google/vit-base-patch16-224-in21k"  # Pre-trained model
  lora_r: 8                    # LoRA rank (lower = more compression)
  lora_alpha: 16               # LoRA scaling factor
  lora_dropout: 0.1            # Dropout for LoRA layers
  freeze_backbone: true        # Keep backbone frozen

# Data configuration
data:
  data_dir: "./data"           # Dataset location
  batch_size: 32               # Batch size for training
  data_split: "non_iid"        # Data distribution: "iid" or "non_iid"
  alpha: 0.5                   # Dirichlet parameter (lower = more heterogeneous)

# Federated learning
federated:
  num_clients: 10              # Total number of clients
  num_rounds: 100              # Number of federated rounds
  client_fraction: 0.3         # Fraction of clients selected per round
  aggregation_method: "fedavg" # Aggregation strategy: "fedavg" or "equal"

# Training hyperparameters
training:
  local_epochs: 5              # Local training epochs per round
  learning_rate: 0.001         # Learning rate for LoRA parameters
  weight_decay: 0.0001         # L2 regularization

# Privacy (optional)
privacy:
  enable_privacy: true         # Enable differential privacy
  epsilon: 2.0                 # Privacy budget (smaller = more private)
  delta: 0.00001              # Privacy parameter for (ε,δ)-DP
  max_grad_norm: 1.0          # Gradient clipping threshold
  secure_aggregation: false    # Enable secure aggregation
```

## Project Structure

```
fedsa_ftl_standalone/
├── src/
│   ├── fedsa_ftl_model.py      # Model architecture with LoRA
│   ├── fedsa_ftl_client.py     # Client-side training logic
│   ├── fedsa_ftl_server.py     # Server aggregation logic
│   ├── data_utils.py           # Data loading and partitioning
│   ├── privacy_utils.py        # Differential privacy mechanisms
│   └── notification_utils.py   # Slack/Discord notifications
├── configs/                     # Configuration files
├── experiments/                 # Output directory for results
├── data/                       # Dataset directory
├── main.py                     # Main training script
├── quickstart.py               # Quick test script
├── notify_completion.py        # Notification handler
└── test_*.py                   # Various test scripts
```

## Output and Results

Training results are saved in `experiments/<experiment_name>/<timestamp>/`:

- `config.yaml` - Configuration used for training
- `results.json` - Complete training history and metrics
- `best_checkpoint.pt` - Best model checkpoint
- `checkpoint_round_N.pt` - Periodic checkpoints

### Metrics Tracked

- Training/test accuracy and loss per round
- Communication cost (MB per round)
- Model compression ratio
- Privacy budget consumption (if DP enabled)
- Per-client performance statistics

## Advanced Features

### Differential Privacy
Enable privacy protection with automatic noise calibration:
```yaml
privacy:
  enable_privacy: true
  epsilon: 2.0  # Total privacy budget
```

### Notification System
Set up Slack/Discord notifications for training completion:
```bash
# Configure notifications
python setup_slack.md  # Follow instructions

# Test notification
python test_slack.py
```

### Custom Models
Supports both Vision Transformer and VGG16 architectures:
- ViT: Better performance, higher computational cost
- VGG16: Lighter weight, faster training

## Performance

- **Communication Reduction**: >100x compared to full model sharing
- **Memory Efficiency**: Only trainable parameters updated
- **Scalability**: Tested with up to 100 clients
- **Convergence**: Typically reaches 90%+ accuracy on CIFAR-10 in 50-100 rounds

## Troubleshooting

### GPU Memory Issues
- Reduce `batch_size` in configuration
- Use VGG16 instead of ViT for lower memory usage

### Slow Training
- Enable GPU: `--gpu 0`
- Reduce `local_epochs` for faster rounds
- Increase `client_fraction` for better convergence

### Data Loading
- Ensure `data_dir` exists and is writable
- First run downloads CIFAR dataset automatically

## Citation

This implementation combines techniques from:
- Feature-based Federated Transfer Learning (FbFTL)
- Federated Learning with Selective Aggregation and LoRA (FedSA-LoRA)