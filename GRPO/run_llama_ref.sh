#!/usr/bin/env bash
# GRPO reference-model server (returns ref per-token logps for the KL term). Port: GRPO_REF_SERVER_PORT (default 59888).
set -euo pipefail
cd "$(dirname "$0")"
# export LLM_DECODER_PATH=/path/to/Llama-3.2-3B
# export GRPO_INIT_CKPT=/path/to/stage3_dpro.ckpt    # Stage-3 D_pro (train_mode=abnorm) = reference policy
# export GRPO_REF_SERVER_PORT=59888
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python3 ref_LlaMa32_server.py
