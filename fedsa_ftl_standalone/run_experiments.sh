#!/bin/bash

python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA-epsilon2.yaml \
  --round 100

python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA-epsilon4.yaml \
  --round 100

python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA-epsilon8.yaml \
  --round 100

python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA-epsilon16.yaml \
  --round 100