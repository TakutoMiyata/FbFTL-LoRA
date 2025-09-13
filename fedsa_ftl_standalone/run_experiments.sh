#!/bin/bash
# Experiment runner script for hyperparameter grid search

# Create experiment directory
mkdir -p experiments/grid_search
mkdir -p logs

# Define experiment configurations
SEEDS=(42 123 456)
LORA_R_VALUES=(8 16)
LOCAL_EPOCHS=(1 2 3)
LR_VALUES=(0.0001 0.0005)
CLIENT_FRACTIONS=(0.3 0.5 1.0)

# Quick test first
echo "Running quick test..."
python quickstart_vit.py --config configs/cifar100_vit_quick_test.yaml > logs/quick_test.log 2>&1
echo "Quick test completed. Check logs/quick_test.log"

# Function to run single experiment
run_experiment() {
    local seed=$1
    local lora_r=$2
    local epochs=$3
    local lr=$4
    local cf=$5
    
    local exp_name="seed${seed}_r${lora_r}_ep${epochs}_lr${lr}_cf${cf}"
    echo "Running experiment: $exp_name"
    
    # Create temporary config
    cat > configs/temp_${exp_name}.yaml <<EOF
seed: $seed
use_gpu: true

experiment:
  name: "grid_search_${exp_name}"
  output_dir: "experiments/grid_search/${exp_name}"

model:
  num_classes: 100
  model_name: "vit_small"
  lora_r: $lora_r
  lora_alpha: 16
  lora_dropout: 0.1
  freeze_backbone: true

data:
  dataset_name: "cifar100"
  data_dir: "./data"
  batch_size: 32
  num_workers: 2
  data_split: "non_iid"
  alpha: 0.5
  model_type: "vit"

federated:
  num_clients: 10
  num_rounds: 50
  client_fraction: $cf
  checkpoint_freq: 10
  aggregation_method: "fedavg"

training:
  local_epochs: $epochs
  optimizer: "adamw"
  learning_rate: $lr
  weight_decay: 0.001

evaluation:
  eval_freq: 5

reproducibility:
  deterministic: false
  benchmark: true
EOF
    
    # Run experiment
    python quickstart_vit.py --config configs/temp_${exp_name}.yaml > logs/${exp_name}.log 2>&1 &
}

# Priority experiments (run sequentially or in parallel based on GPU memory)
echo "Starting priority experiments..."

# Priority 1: Effect of local epochs (with fixed settings)
for epochs in "${LOCAL_EPOCHS[@]}"; do
    run_experiment 42 8 $epochs 0.0005 0.3
    wait  # Wait for completion before next (remove for parallel execution)
done

# Priority 2: Effect of LoRA rank
for lora_r in "${LORA_R_VALUES[@]}"; do
    run_experiment 42 $lora_r 2 0.0005 0.3
    wait
done

# Priority 3: Effect of learning rate
for lr in "${LR_VALUES[@]}"; do
    run_experiment 42 8 2 $lr 0.3
    wait
done

# Priority 4: Effect of client fraction
for cf in "${CLIENT_FRACTIONS[@]}"; do
    run_experiment 42 8 2 0.0005 $cf
    wait
done

# Full grid search (optional - resource intensive)
# Uncomment below for full grid search
# for seed in "${SEEDS[@]}"; do
#     for lora_r in "${LORA_R_VALUES[@]}"; do
#         for epochs in "${LOCAL_EPOCHS[@]}"; do
#             for lr in "${LR_VALUES[@]}"; do
#                 run_experiment $seed $lora_r $epochs $lr 0.3
#             done
#         done
#     done
# done

echo "All experiments completed. Check logs/ directory for outputs."

# Clean up temporary configs
rm -f configs/temp_*.yaml