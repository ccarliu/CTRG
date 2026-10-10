#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Override asset paths here if your copies live elsewhere (see paths_config.py)
# ---------------------------------------------------------------------------
# export BIOMEDVLP_BERT=/path/to/BiomedVLP_cxr_bert
# export CTRG_ANNOTATION_JSON=/path/to/CTRG/label/chest/annotation.json
# export TEXT_LATENT_NPZ=/path/to/text_latent_feature_*.npz
# export PRETRAIN_CKPT=""   # empty => train from scratch

# GPUs to use
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python main_3D.py with \
    task_pretrain_m3ae_3D \
    per_gpu_batchsize=1 \
    num_gpus=1 \
    num_nodes=1 \
    num_workers=4 \
    precision=16 \
    log_dir=results \
    default_root_dir=results \
    "$@"
