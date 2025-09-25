#!/bin/bash

# 1つ目のジョブを実行
nohup python quickstart_resnetpy \
  --config configs/experiment_configs_iid/IID-FedAvg.yaml \
  --round 100

# 上が終了したら次を実行
nohup python quickstart_resnet.py \
  --config configs/experiment_configs_iid/IID-DP-FedSA-LoRA.yaml \
  --round 100