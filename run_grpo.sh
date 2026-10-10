#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# ---------------------------------------------------------------------------
# Stage 4: GRPO client (policy D_pro). Start the servers under GRPO/ first:
#   GRPO/run_llama_ref.sh, GRPO/run_llama_ref_think.sh, GRPO/run_F1_reward.sh
# ---------------------------------------------------------------------------
# export GRPO_INIT_CKPT=/path/to/stage3_dpro.ckpt            # initial policy (D_pro)
# export GRPO_REF_SERVER_URL=http://localhost:59888
# export GRPO_REWARD_SERVER_URL=http://localhost:60010
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python main_report_gen_grpo.py with \
    task_finetune_rg \
    per_gpu_batchsize=1 \
    num_gpus=1 \
    num_nodes=1 \
    num_workers=4 \
    precision=16 \
    log_dir=results \
    default_root_dir=results \
    "$@"
