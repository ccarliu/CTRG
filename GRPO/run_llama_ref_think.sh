#!/usr/bin/env bash
# GRPO reward generation + aggregation server (D_full generates the report, then
# scores it via F1/GREEN). This is what the client's reward_think() calls.
# Port: GRPO_REWARD_SERVER_PORT (default 60010).
set -euo pipefail
cd "$(dirname "$0")"
# export BIOMEDVLP_BERT=/path/to/BiomedVLP_cxr_bert
# export GRPO_DFULL_CKPT=/path/to/stage3_dfull.ckpt   # Stage-3 D_full (train_mode=full)
# export GRPO_REWARD_SERVER_PORT=60010
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} python3 ref_LlaMa32_server_gen.py \
  with data_root=data/pretrain_arrows/ \
  num_gpus=1 num_nodes=1 \
  task_finetune_rg \
  per_gpu_batchsize=1 \
  "$@"
