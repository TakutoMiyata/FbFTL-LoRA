# FedSA-FTL: Federated Share-A Transfer Learning

This is a standalone implementation of FedSA-FTL, which combines:
- **Frozen backbone architecture** from Feature-based Federated Transfer Learning (FbFTL)
- **Selective A-matrix aggregation** from FedSA-LoRA

## Overview

FedSA-FTL is a novel federated learning approach that:
1. Uses a frozen pre-trained backbone (e.g., ViT) as a feature extractor
2. Applies LoRA adaptation only to the task-specific head
3. Communicates only LoRA A-matrices to the server (reducing communication by >100x)
4. Keeps B-matrices local for client personalization

## Architecture

```
┌─────────────────────────┐
│   Pre-trained ViT       │ ← Frozen (no gradients)
│   (Feature Extractor)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Classification Head    │ ← LoRA-adapted
│   W = W₀ + B·A          │   (A: shared, B: personalized)
└─────────────────────────┘
```

## Installation

```bash
# Clone the repository
cd fedsa_ftl_standalone

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run a small-scale test to verify the implementation:

```bash
python quickstart.py
```

This will:
- Use 3 clients with non-IID CIFAR-10 data
- Run 5 federated rounds
- Save results to `experiments/quickstart/`

## Full Experiment

Run a full experiment with custom configuration:

```bash
# Basic experiment
python main.py --config configs/cifar10_vit_base.yaml

# Challenging non-IID setting
python main.py --config configs/cifar10_vit_challenging.yaml

# With custom seed and GPU
python main.py --config configs/cifar10_vit_base.yaml --seed 123 --gpu 0
```

## Configuration

Key configuration parameters in YAML files:

```yaml
model:
  lora_r: 8           # LoRA rank (lower = more compression)
  lora_alpha: 16      # LoRA scaling factor
  freeze_backbone: true  # Freeze pre-trained backbone

data:
  data_split: "non_iid"  # Data distribution
  alpha: 0.5            # Dirichlet parameter (lower = more heterogeneous)

federated:
  num_clients: 10       # Number of clients
  num_rounds: 100       # Number of federated rounds
  client_fraction: 0.3  # Fraction of clients per round

training:
  local_epochs: 5       # Local training epochs
  learning_rate: 1e-3   # Learning rate for LoRA parameters
```

## Project Structure

```
fedsa_ftl_standalone/
├── src/
│   ├── fedsa_ftl_model.py    # Model with frozen backbone + LoRA
│   ├── fedsa_ftl_client.py   # Client with local training
│   ├── fedsa_ftl_server.py   # Server with A-matrix aggregation
│   └── data_utils.py         # Data loading and splitting
├── configs/
│   ├── cifar10_vit_base.yaml       # Standard configuration
│   └── cifar10_vit_challenging.yaml # Challenging non-IID setting
├── experiments/              # Output directory for results
├── main.py                   # Main training script
├── quickstart.py            # Quick test script
└── requirements.txt         # Python dependencies
```

## Key Features

1. **Communication Efficiency**: Only LoRA A-matrices are communicated (typically <1MB per round)
2. **Personalization**: B-matrices remain local for client-specific adaptation
3. **Transfer Learning**: Leverages pre-trained models for better performance
4. **Non-IID Support**: Handles heterogeneous data distributions

## Results

After training, results are saved in the experiment directory:
- `config.yaml`: Configuration used
- `results.json`: Training history and metrics
- `best_model.pt`: Best model checkpoint
- `checkpoint_round_N.pt`: Periodic checkpoints

## Citation

This implementation combines ideas from:
- Feature-based Federated Transfer Learning (FbFTL)
- Selective Aggregation for Low-Rank Adaptation (FedSA-LoRA)