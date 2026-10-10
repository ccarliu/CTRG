#!/usr/bin/env bash
# GRPO F1 reward server (RadBERT abnormality classifier -> CE-F1). OPTIONAL reward.
# Port: GRPO_F1_SERVER_PORT (default 61000).
set -euo pipefail
cd "$(dirname "$0")"
# export RADBERT_PATH=/path/to/RadBERT
# export RADBERT_CLASSIFIER_CKPT=/path/to/RadBertClassifier.pth
# export GRPO_F1_SERVER_PORT=61000
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python reward_server.py
