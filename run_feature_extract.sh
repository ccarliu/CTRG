#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Stage 1.5: visual feature extraction / caching (CT-ViT perorgan encoder).
# Runs the Stage-1 pretrained encoder over all volumes and caches
# selected_patch / image_feature / local_text / local_image .npy to
# FEATURE_STORE_PATH, which later stages read as `feature_path`.
# ---------------------------------------------------------------------------
# export BIOMEDVLP_BERT=/path/to/BiomedVLP_cxr_bert
# export CTRG_ANNOTATION_JSON=/path/to/CTRG/label/chest/annotation.json
# export TEXT_LATENT_NPZ=/path/to/text_latent_feature_*.npz
# Stage-1 encoder checkpoint used to extract features:
# export PRETRAIN_CKPT=/path/to/stage1_pretrain.ckpt
# Output cache dir (must equal the dataset's feature_path):
# export FEATURE_STORE_PATH=/path/to/img_feature/version_628_e5_train
# Which split to extract: train | val | test
export EXTRACT_SPLIT=${EXTRACT_SPLIT:-train}

num_gpus=${NUM_GPUS:-3}
master_port=${MASTER_PORT:-29601}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

torchrun --nproc_per_node=${num_gpus} --master_port=${master_port} \
    main_feature_extract.py with \
    task_pretrain_m3ae_3D \
    num_gpus=${num_gpus} num_nodes=1 \
    per_gpu_batchsize=1 \
    num_workers=4 \
    "$@"
