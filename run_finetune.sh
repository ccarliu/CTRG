#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Override asset paths here if your copies live elsewhere (see paths_config.py)
# ---------------------------------------------------------------------------
# export BIOMEDVLP_BERT=/path/to/BiomedVLP_cxr_bert
# export CTRG_ANNOTATION_JSON=/path/to/CTRG/label/chest/annotation.json
# export LLM_DECODER_PATH=/path/to/Llama-3.2-3B      # causal-LM report decoder
# export MED_CONFIG_BLIP=/path/to/med_config_blip.json   # defaults to repo copy
# export FINETUNE_INIT_CKPT=/path/to/stage1.ckpt     # Stage-1 encoder (or full Stage-3 ckpt with test_only=True)
# export SELECTOR_INIT_CKPT=/path/to/stage2.ckpt     # Stage-2 AP-LTF selector (needed for train_mode=full)
# Choose the model with: bash run_finetune.sh train_mode=full   (D_full)
#                        bash run_finetune.sh train_mode=abnorm (D_pro)
# Optional (only used by GRPO reward / retrieval helpers):
# export BLEU_METRIC_PATH=/path/to/metric_cal/bleu/bleu.py
# export ORGAN_CENTERS_NPY=/path/to/text_embedding_organ_centers_train.npy
# export REGION_REPORT_CSV=/path/to/train_region_report.csv

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python main_report_gen_rate.py with \
    task_finetune_rg \
    per_gpu_batchsize=1 \
    num_gpus=1 \
    num_nodes=1 \
    num_workers=4 \
    precision=16 \
    log_dir=results \
    default_root_dir=results \
    "$@"
