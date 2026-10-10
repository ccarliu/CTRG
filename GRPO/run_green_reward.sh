#!/usr/bin/env bash
# GRPO GREEN reward server (GREEN model). OPTIONAL reward (alternative to F1).
# Port: GRPO_GREEN_SERVER_PORT (default 59879).
set -euo pipefail
cd "$(dirname "$0")"
# export GREEN_MODEL_PATH=/path/to/Green
# export GRPO_GREEN_SERVER_PORT=59879
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3,2,1} python3 reward_server_green_v3.py
