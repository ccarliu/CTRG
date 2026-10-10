# Mimicking Radiologists: A Coarse-to-Fine Framework with Structural Sparse Tokens for Dual-LLM Computed Tomography Report Generation

**ECCV 2026** · [Paper (Springer)](https://doi.org/10.1007/978-3-032-37531-5_21)

Hong Liu, Dong Wei, Yefeng Zheng, Xian Wu, Liansheng Wang

Xiamen University · Tencent Jarvis Lab · Westlake University

This is the official code of the ECCV 2026 paper above. The framework follows the
coarse-to-fine visual search of radiologists:

1. A ViT-based image-text alignment encoder extracts a **global token** and **local patch
   tokens** for each anatomical structure, trained with a **mask-guided sparse
   negative-entropy loss** (`L_sparse`).
2. An **abnormality-proposal LLM** (`D_pro`) reads the global tokens and clinical metadata
   and proposes a shortlist of candidate abnormalities.
3. The **abnormality-prompted local token filter (AP-LTF)** uses the shortlist to keep the
   most informative patch tokens per structure.
4. A **report-generation LLM** (`D_full`) writes the full report from the global token, the
   selected local tokens and clinical metadata. During training, ground-truth
   abnormalities are prefixed to the reference report, and **GRPO** aligns `D_pro` with
   `D_full`.

This repository covers the CT-RATE pipeline. If you use it, please cite the paper
(see [Citation](#citation)).

## Pipeline

The code implements the method as four training stages:

- **Stage 1 — Encoder pretraining** — `main_3D.py` trains
  `M3AETransformerSS_3D_lmae_rg_pretrain_v29_Qwen_perorgan_mask`
  (CT-ViT + structure queries + image-text alignment + `L_sparse`) on raw CT volumes.
- **Stage 1.5 — Feature extraction / caching** — `main_feature_extract.py` runs the
  Stage-1 encoder over all volumes and caches
  `selected_patch / image_feature / local_text / local_image .npy` per case into
  `FEATURE_STORE_PATH`. Later stages read these cached features instead of
  re-running the encoder. Multi-GPU via `torchrun` (static sharding, resumable).
- **Stage 2 — Feature selection (AP-LTF / post-alignment)** — `main_feature_select.py`
  trains `M3AETransformerSS_3D_lmae_rg_pretrain_v29_Qwen_perorgan_postalign`
  (the `DefaultSelector_2` AP-LTF + per-organ alignment), warm-started from Stage 1.
- **Stage 3 — Decoder finetuning (report generation)** — `main_report_gen_rate.py`
  trains `M3AETransformerSS_3D_lmae_rg_v38_llm_v3` (Llama-3.2-3B + LoRA).
  `train_mode=full` trains the full report model `D_full` (uses the Stage-2 selector);
  `train_mode=abnorm` trains the abnormality-proposal model `D_pro`. The two runs are
  independent and can be done in any order.
- **Stage 4 — GRPO collaborative refinement** — `main_report_gen_grpo.py` refines
  `D_pro` with GRPO. The frozen reference model, `D_full` (report generation + reward
  aggregation) and the reward model (F1 or GREEN) run as separate HTTP servers under
  `GRPO/`. Start the servers first, then the client.

Stages 1 and 1.5 use the raw-image `MedicatDataModule_3D_RATE_hr` pipeline; Stages 2-4
use `MedicatDataModule_3D_RATE_hr_fea`, which reads the features cached in Stage 1.5.

## Layout

```
ctrg_pretrain/
├── main_3D.py                 # Stage 1: encoder pretraining
├── main_feature_extract.py    # Stage 1.5: feature extraction / caching (torchrun)
├── main_feature_select.py     # Stage 2: feature selection (AP-LTF / post-align)
├── main_report_gen_rate.py    # Stage 3: decoder finetuning (D_full / D_pro) + evaluation
├── main_report_gen_grpo.py    # Stage 4: GRPO client (policy D_pro)
├── run_pretrain.sh / run_feature_extract.sh / run_feature_select.sh
├── run_finetune.sh / run_grpo.sh           # launch scripts; extra args are passed to sacred
├── classifier.py              # RadBertClassifier (F1 server)
├── GRPO/                      # Stage-4 servers (HTTP)
│   ├── ref_LlaMa32_server.py        # reference model (D_pro init), port 59888
│   ├── ref_LlaMa32_server_gen.py    # D_full report generation + reward aggregation, port 60010
│   ├── reward_server.py             # F1 server (RadBERT), port 61000
│   ├── reward_server_green_v3.py    # GREEN reward server, port 59879
│   └── run_*.sh
├── data_preprocess/           # Stage 0: CT-RATE nii.gz -> npz (see its README)
├── tools/pack_data.py         # (maintainers) build / pack the data bundle
├── third_party/GREEN/         # vendored, modified GREEN (see NOTICE.md)
├── assets/                    # small bundled assets (configs, tokenizer, npz, bleu)
├── data/                      # the data bundle, extracted here (not in git)
├── paths_config.py            # all external paths (env-overridable)
├── ct_clip/  ctvit/           # CT-CLIP / CT-ViT (imported top-level)
└── ctrg/                      # config, datasets, datamodules, models, losses, metrics
```

## Reproduction

### 0. Hardware and disk

- The authors used Tesla V100 32 GB GPUs; every stage runs with batch size 1 per GPU.
- Stage 1.5 defaults to 3 GPUs (`NUM_GPUS`). Stage 4 needs at least 4 GPUs
  (client, reference server, D_full server, F1 server); the GREEN server uses 3 more
  by default.
- Disk: CT-RATE raw + preprocessed volumes are several TB; the Stage-1.5 feature cache
  is ~411 GB for the train split; the data bundle is 1.1 GB compressed / ~4 GB extracted.

### 1. Environment

Main environment (all stages):

```bash
conda create -n ctrg python=3.8 -y && conda activate ctrg
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"     # needed at import time
# Java 8 is required by the METEOR metric (jars are bundled in ctrg/generation_api):
#   e.g. sudo apt-get install openjdk-8-jre
```

`requirements.txt` pins the versions used by the authors (Llama-3.2 requires
`transformers>=4.43`).

GREEN environment (only for the optional GREEN reward in Stage 4; `third_party/GREEN`
requires Python 3.12.1 exactly):

```bash
conda create -n ctrg_green python=3.12.1 -y && conda activate ctrg_green
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e third_party/GREEN          # the vendored, modified GREEN (not the PyPI one)
pip install pandas bottle tornado requests
```

### 2. Downloads

| What | Where | Set |
|---|---|---|
| Data bundle `ctrg_data.tar.gz` (reports, labels, region reports, metadata, report trees, fast masks, text embeddings) | [released with this repo](https://drive.google.com/file/d/112wCts4_APDUwS3jyzp4Ffps6s8sslCX/view?usp=sharing) | extract in the repo root (`tar -xzf ctrg_data.tar.gz` → `./data`) or `CTRG_DATA_ROOT` |
| BiomedVLP-CXR-BERT | HF `microsoft/BiomedVLP-CXR-BERT-specialized` | `BIOMEDVLP_BERT` |
| CT-CLIP `CT_CLIP_zeroshot.pt` (Stage 2, text branch only) | CT-CLIP release | `CT_CLIP_CKPT` |
| Llama-3.2-3B (Stages 3-4, gated) | HF `meta-llama/Llama-3.2-3B` | `LLM_DECODER_PATH` |
| RadBERT + `RadBertClassifier.pth` (CE-F1 in Stage 3 evaluation, F1 reward in Stage 4) | RadBERT from HF; classifier head released with this repo | `RADBERT_PATH`, `RADBERT_CLASSIFIER_CKPT` |
| GREEN-RadLlama2-7b (optional, Stage 4) | HF `StanfordAIMI/GREEN-RadLlama2-7b` | `GREEN_MODEL_PATH` |
| CT-RATE volumes (Stages 1 and 1.5 only, gated) | HF `ibrahimhamamci/CT-RATE` | `CTRATE_RAW_TRAIN_DIR`, `CTRATE_RAW_VALID_DIR` |

The roberta-base text config/tokenizer (`CTRG_MODEL_PATH`), the CTRG `annotation.json`,
the text-latent npz and the BLEU script are already in `assets/`.

### 3. Preprocess CT-RATE (Stage 0; needed for Stages 1 and 1.5 only)

```bash
export CTRATE_RAW_TRAIN_DIR=/data/CT-RATE/dataset/train
export CTRATE_RAW_VALID_DIR=/data/CT-RATE/dataset/valid
export CTRATE_TRAIN_IMAGES=/data/ctrate_pre/train_preprocessed
export CTRATE_VALID_IMAGES=/data/ctrate_pre/valid_preprocessed
python data_preprocess/preprocess_ctrate_train.py
python data_preprocess/preprocess_ctrate_valid.py
python data_preprocess/datacheck.py        # optional: remove unreadable npz
```

The metadata CSVs come from the data bundle. Structural masks are not regenerated: the
bundled `mask_fast` covers every case.

### 4. Train

Checkpoints are written to `results/<exp_name>-seed0-from_/version_N/checkpoints/`.
Each stage reads the previous stage's checkpoint from an env var. All `run_*.sh` pass
extra arguments to sacred (e.g. `bash run_finetune.sh train_mode=abnorm`).

```bash
# Stage 1: encoder pretraining
export PRETRAIN_CKPT=""                    # "" = from scratch
bash run_pretrain.sh
export PRETRAIN_CKPT=results/.../version_A/checkpoints/<best>.ckpt

# Stage 1.5: cache encoder features (train and val into the same directory)
export FEATURE_STORE_PATH=/data/ctrg_features
EXTRACT_SPLIT=train bash run_feature_extract.sh
EXTRACT_SPLIT=val   bash run_feature_extract.sh

# Stage 2: AP-LTF feature selection
export FEATSELECT_INIT_CKPT=$PRETRAIN_CKPT
bash run_feature_select.sh                 # -> Stage-2 ckpt (version_B)
```

Stage 3 validation computes CE-F1 by sending generated reports to the F1 server, so
start it before training or evaluating:

```bash
# F1 server (RadBERT classifier), keep it running
CUDA_VISIBLE_DEVICES=1 bash GRPO/run_F1_reward.sh

# Stage 3: D_pro and D_full, independent runs
export FINETUNE_INIT_CKPT=$PRETRAIN_CKPT                 # Stage-1 encoder
export SELECTOR_INIT_CKPT=results/.../version_B/checkpoints/<best>.ckpt   # Stage-2 selector
bash run_finetune.sh train_mode=abnorm     # D_pro  -> version_C
bash run_finetune.sh train_mode=full       # D_full -> version_D
```

Stage 4 (optional):

```bash
export GRPO_INIT_CKPT=results/.../version_C/checkpoints/<best>.ckpt    # D_pro
export GRPO_DFULL_CKPT=results/.../version_D/checkpoints/<best>.ckpt   # D_full
CUDA_VISIBLE_DEVICES=1 bash GRPO/run_llama_ref.sh        # reference model (port 59888)
CUDA_VISIBLE_DEVICES=2 bash GRPO/run_llama_ref_think.sh  # D_full + reward aggregation (60010)
CUDA_VISIBLE_DEVICES=3 bash GRPO/run_F1_reward.sh        # F1 reward (61000)
#  or, in the ctrg_green env: bash GRPO/run_green_reward.sh   (GREEN, 59879)
CUDA_VISIBLE_DEVICES=0 bash run_grpo.sh                  # GRPO client
```

If the servers run on other hosts, set `GRPO_REF_SERVER_URL`, `GRPO_REWARD_SERVER_URL`,
`GRPO_F1_SERVER_URL`, `GRPO_GREEN_SERVER_URL`.

### 5. Evaluate

With the F1 server running:

```bash
FINETUNE_INIT_CKPT=results/.../version_D/checkpoints/<best>.ckpt \
  bash run_finetune.sh train_mode=full test_only=True
```

Generated reports are appended to `results/predictions/test_<train_mode>_<ckpt>.csv`
(validation during training: `results/predictions/val_<train_mode>.csv`), one row per
case: `data_name, abnormality labels, generated report, reference report`.

During validation/test only **CE-F1** is computed online (via the F1 server). It is
logged to TensorBoard under `rg/<phase>/BLEU_1` (historical name) and drives checkpoint
selection (`val/the_metric`). NLG metrics (BLEU, METEOR, ROUGE-L, ...) are computed
offline from the prediction CSV.

Use `fast_dev_run=True` (e.g. `bash run_finetune.sh fast_dev_run=True`) for a quick
smoke test of any stage.

## Configuration reference

All paths live in `paths_config.py` and can be overridden with environment variables.
Assets in `assets/` and the data bundle default to locations inside the repo; weights,
checkpoints, CT volumes and features have no default (each entry point checks the ones
its stage needs and prints what to `export` if they are missing), so they must be
set in a new environment.

| Env var | Purpose | Stage |
|---|---|---|
| `CTRG_DATA_ROOT` | root of the data bundle (default `<repo>/data`) | all |
| `BIOMEDVLP_BERT` | BiomedVLP CXR-BERT dir | all |
| `CTRG_MODEL_PATH` | roberta-base config + tokenizer (bundled at `assets/ctrg_text_config`) | all |
| `CTRG_ANNOTATION_JSON` | CTRG chest `annotation.json`, read at import time (bundled) | all |
| `TEXT_LATENT_NPZ` | text latent features for retrieval accuracy (bundled) | 1, 2 |
| `CTRATE_RAW_{TRAIN,VALID}_DIR` | CT-RATE nii.gz | 0 |
| `CTRATE_{TRAIN,VALID}_METADATA` | spacing / rescale CSVs (bundle) | 0 |
| `CTRATE_{TRAIN,VALID}_IMAGES` | preprocessed CT volumes | 1, 1.5 |
| `CTRATE_{TRAIN,VALID}_MASK_FAST` | fast structural masks (bundle) | 1, 1.5 |
| `TEXT_EMBEDDING_DIR` | per-case text embeddings (bundle) | 1, 1.5 |
| `CTRATE_{TRAIN,VALID}_REPORTS` / `_LABELS` / `_REGION_REPORT` | report / label / region-report CSVs (bundle) | all |
| `JSON_PATH_{TRAIN,VALID}` | per-case report hierarchy trees: abnormality and prior text (bundle) | all |
| `PRETRAIN_CKPT` | Stage-1 warm start; encoder used by Stage 1.5 | 1, 1.5 |
| `FEATURE_STORE_PATH` | Stage-1.5 output dir; default for `FEATURE_PATH_{TRAIN,VALID}` | 1.5-4 |
| `EXTRACT_SPLIT` | split to extract: `train` / `val` / `test` | 1.5 |
| `CT_CLIP_CKPT` | `CT_CLIP_zeroshot.pt`; its text branch inits the frozen text encoder | 2 |
| `FEATSELECT_INIT_CKPT` | Stage-2 warm start (Stage-1 ckpt) | 2 |
| `LLM_DECODER_PATH` | Llama-3.2-3B | 3, 4 |
| `MED_CONFIG_BLIP` | BLIP med config json (bundled) | 3, 4 |
| `FINETUNE_INIT_CKPT` | Stage-3 init (Stage-1 ckpt); full Stage-3 ckpt with `test_only=True` | 3 |
| `SELECTOR_INIT_CKPT` | Stage-2 ckpt providing the `selector.*` (AP-LTF) weights | 3 |
| `RADBERT_PATH` / `RADBERT_CLASSIFIER_CKPT` | RadBERT + classifier head for the F1 server | 3 (eval), 4 |
| `GRPO_INIT_CKPT` | Stage-3 D_pro: GRPO initial policy and reference model | 4 |
| `GRPO_DFULL_CKPT` | Stage-3 D_full used by the reward generation server | 4 |
| `GRPO_DFULL_GROUP_SIZE` / `GRPO_DFULL_K_CUR` | AP-LTF candidates / kept patches in the D_full server (250 / 15) | 4 |
| `GREEN_MODEL_PATH` | GREEN model | 4 (GREEN) |
| `BLEU_METRIC_PATH` | BLEU script for `evaluate.load` (bundled) | 4 |
| `GRPO_{REF,REWARD,F1,GREEN}_SERVER_URL` / `_PORT` | server endpoints (59888 / 60010 / 61000 / 59879) | 3 (F1), 4 |
| `PREPROCESS_NUM_WORKERS` | preprocessing pool size | 0 |

## Data bundle (`$CTRG_DATA_ROOT`)

```
data/
├── ctrate/
│   ├── reports/                 train / validation report CSVs
│   ├── labels/                  train / valid predicted abnormality labels
│   ├── region_report/           per-structure abnormality region reports
│   ├── metadata/                spacing / rescale metadata (preprocessing)
│   ├── report_hierarchy_tree/{train,valid}/<case>.json
│   └── mask_fast/{train,valid}/<case>.npy
└── text_embedding/ipmiversion/<case>_text_embedding.npy
```

Not included: CT-RATE volumes (download + `data_preprocess/`) and the Stage-1.5 feature
cache (produced by Stage 1.5). The bundle contains data derived from CT-RATE and
RadGenome-ChestCT and is subject to their (non-commercial) terms.

Maintainers build it with
`python tools/pack_data.py --ctrate-src <CT-RATE src root> --m3ae-src <dir with text_embedding/>`
(symlinks into `./data`), `--check` (verify) and `--tar ctrg_data.tar.gz` (archive for upload).

## Notes

- Other dataset classes in `ctrg/datasets/pretraining_ctrg_dataset.py`
  (`*_clip`, `*_all`, `*_dpo`, `CTRGDataset`, `CTRGDataset_RATE`, ...) are not used by
  any entry point and still contain their original absolute paths.
- `third_party/GREEN` is a modified copy of Stanford-AIMI/GREEN (MIT); see its
  `NOTICE.md` and `ctrg_modifications.patch`.

## Citation

If you find this code useful, please cite:

```bibtex
@InProceedings{10.1007/978-3-032-37531-5_21,
author="Liu, Hong
and Wei, Dong
and Zheng, Yefeng
and Wu, Xian
and Wang, Liansheng",
editor="Favaro, Paolo
and Kukelova, Zuzana
and Maki, Atsuto
and Rohrbach, Anna
and Schindler, Konrad
and Tombari, Federico",
title="Mimicking Radiologists: A Coarse-to-Fine Framework with Structural Sparse Tokens for Dual-LLM Computed Tomography Report Generation",
booktitle="Computer Vision -- ECCV 2026",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="376--393",
abstract="Computed tomography report generation (CTRG) automates radiology reporting to reduce clinical workload and facilitate patient care. Recent efforts in applying the rapidly developing large language models (LLMs) have advanced the field; yet, they still face a fundamental challenge due to the large volume of 3D data: effectively reducing high feature redundancy and computational burden while simultaneously extracting information-rich representations. To address this challenge, this work presents a novel CTRG framework that fully imitates the coarse-to-fine visual search pipeline practiced by radiologists. Our framework first employs a ViT-based image-text alignment architecture to extract a global token and local patch tokens for each anatomical structure, enhanced by a mask-guided sparse negative-entropy loss. An abnormality-proposal LLM then processes the global token and clinical metadata to propose a shortlist of candidate abnormalities. This shortlist prompts an abnormality-prompted local token filter (AP-LTF) to select the most informative patch tokens, effectively reducing redundancy while preserving critical information. Finally, a report-generation LLM takes in the global token, selected local tokens, and clinical metadata to compose a full report. During training, ground-truth abnormalities are prefixed to the reference report to enhance awareness of clinic-relevant findings, and group relative policy optimization (GRPO) aligns the abnormality-proposal and report-generation LLMs for collaborative efficacy. Experimental results on two public CTRG datasets demonstrate the superior performance of our framework compared to existing state-of-the-art methods in terms of clinical efficacy, RaTEScore, and GREEN scores. Ablation studies further validate the effectiveness of its novel design.",
isbn="978-3-032-37531-5"
}
```
