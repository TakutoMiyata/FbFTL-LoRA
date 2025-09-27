#!/bin/bash

# 上が終了したら次を実行
nohup python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA-epsilon2.yaml \
  --round 100 &

# 上が終了したら次を実行
nohup python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA-epsilon4.yaml \
  --round 100 &

# 上が終了したら次を実行
nohup python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA-epsilon8.yaml \
  --round 100 &

# 上が終了したら次を実行
nohup python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA-epsilon16.yaml \
  --round 100 &

