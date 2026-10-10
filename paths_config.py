"""
Centralized, environment-driven paths for the CTRG repo.

Two kinds of entries:

* Repo-internal (assets/, the data bundle under $CTRG_DATA_ROOT, configs, server
  ports): have working defaults, nothing to set.
* User-provided (model weights, checkpoints, CT volumes, feature cache): have NO
  default. Set them with `export NAME=...` (e.g. in the run_*.sh scripts). Each
  entry point calls `require(...)` for the variables its stage needs and stops with
  a list of what is missing. See README "Reproduction" for where to get each item.
"""
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_REPO_ROOT, "assets")

_HELP = {}


def _user(name, help_text):
    """User-provided path: read from the environment, empty if unset."""
    _HELP[name] = help_text
    return os.environ.get(name, "")


def require(*names):
    """Raise with a readable message if any of the given variables is unset."""
    g = globals()
    missing = [n for n in names if not g.get(n)]
    if missing:
        lines = "\n".join(f"  export {n}=...    # {_HELP.get(n, '')}" for n in missing)
        raise RuntimeError(
            "Missing required path(s); set them before running "
            "(see paths_config.py / README 'Reproduction'):\n" + lines
        )


# ============================================================================
# Pretrained models (download)
# ============================================================================
BIOMEDVLP_BERT = _user(
    "BIOMEDVLP_BERT",
    "dir of HF microsoft/BiomedVLP-CXR-BERT-specialized (all stages)",
)
CT_CLIP_CKPT = _user(
    "CT_CLIP_CKPT",
    "file CT_CLIP_zeroshot.pt from the CT-CLIP release; only its text branch is "
    "used, to init the frozen text encoder (Stage 2)",
)
LLM_DECODER_PATH = _user(
    "LLM_DECODER_PATH",
    "dir of HF meta-llama/Llama-3.2-3B, the report decoder (Stages 3-4)",
)
RADBERT_PATH = _user(
    "RADBERT_PATH",
    "dir of the RadBERT model, base of the F1 server (Stage-3 eval CE-F1, Stage-4 F1 reward)",
)
RADBERT_CLASSIFIER_CKPT = _user(
    "RADBERT_CLASSIFIER_CKPT",
    "file RadBertClassifier.pth, the abnormality classifier head (released with this repo)",
)
GREEN_MODEL_PATH = _user(
    "GREEN_MODEL_PATH",
    "dir of HF StanfordAIMI/GREEN-RadLlama2-7b (optional GREEN reward, Stage 4)",
)

# ============================================================================
# Checkpoints produced by the pipeline (or released by the authors)
# ============================================================================
PRETRAIN_CKPT = _user(
    "PRETRAIN_CKPT",
    "Stage-1 encoder .ckpt (results/.../checkpoints/*.ckpt from main_3D.py). "
    "Stage 1: optional warm start (empty = from scratch); Stage 1.5: required",
)
FEATSELECT_INIT_CKPT = _user(
    "FEATSELECT_INIT_CKPT",
    "Stage-1 encoder .ckpt to warm-start Stage 2 (usually = PRETRAIN_CKPT; empty = from scratch)",
)
FINETUNE_INIT_CKPT = _user(
    "FINETUNE_INIT_CKPT",
    "Stage 3 training: Stage-1 encoder .ckpt; with test_only=True: the Stage-3 "
    "(D_full / D_pro) .ckpt to evaluate",
)
SELECTOR_INIT_CKPT = _user(
    "SELECTOR_INIT_CKPT",
    "Stage-2 .ckpt (main_feature_select.py) providing the selector.* (AP-LTF) "
    "weights; needed when training D_full (train_mode=full)",
)
GRPO_INIT_CKPT = _user(
    "GRPO_INIT_CKPT",
    "Stage-3 D_pro .ckpt (train_mode=abnorm): GRPO initial policy and reference model (Stage 4)",
)
GRPO_DFULL_CKPT = _user(
    "GRPO_DFULL_CKPT",
    "Stage-3 D_full .ckpt (train_mode=full), loaded by the reward generation server (Stage 4)",
)
# AP-LTF selection used by D_full inside the reward server: candidate patches per
# structure (Group_s) and kept patches (k_cur); values used in the paper.
GRPO_DFULL_GROUP_SIZE = int(os.environ.get("GRPO_DFULL_GROUP_SIZE", "250"))
GRPO_DFULL_K_CUR = int(os.environ.get("GRPO_DFULL_K_CUR", "15"))

# ============================================================================
# Large data (not in the data bundle)
# ============================================================================
# Raw CT-RATE volumes (HF ibrahimhamamci/CT-RATE) -> input of data_preprocess/
CTRATE_RAW_TRAIN_DIR = _user(
    "CTRATE_RAW_TRAIN_DIR",
    "CT-RATE raw train *.nii.gz root (dataset/train), input of preprocess_ctrate_train.py",
)
CTRATE_RAW_VALID_DIR = _user(
    "CTRATE_RAW_VALID_DIR",
    "CT-RATE raw valid *.nii.gz root (dataset/valid), input of preprocess_ctrate_valid.py",
)
# Preprocessed volumes (*.npz): output of data_preprocess/, input of Stages 1 and 1.5
CTRATE_TRAIN_IMAGES = _user(
    "CTRATE_TRAIN_IMAGES",
    "dir for preprocessed train volumes (*.npz): written by preprocess_ctrate_train.py, "
    "read by Stages 1 / 1.5",
)
CTRATE_VALID_IMAGES = _user(
    "CTRATE_VALID_IMAGES",
    "dir for preprocessed valid volumes (*.npz): written by preprocess_ctrate_valid.py, "
    "read by Stages 1 / 1.5",
)
# Optional: structural segmentation preprocessing (NOT needed; mask_fast is bundled)
CTRATE_RAW_SEG_DIR = _user(
    "CTRATE_RAW_SEG_DIR",
    "raw segmentation *.nii.gz dir, input of the optional preprocess_ctrate_mask.py",
)
CTRATE_MASK_PREPROCESSED = _user(
    "CTRATE_MASK_PREPROCESSED",
    "output dir of the optional preprocess_ctrate_mask.py",
)
PREPROCESS_NUM_WORKERS = int(os.environ.get("PREPROCESS_NUM_WORKERS", "10"))

# Cached encoder features: written by Stage 1.5, read by Stages 2-4
FEATURE_STORE_PATH = _user(
    "FEATURE_STORE_PATH",
    "dir for the Stage-1.5 feature cache (~411 GB for train; train and val go to the "
    "same dir); written by main_feature_extract.py, read by Stages 2-4",
)
FEATURE_PATH_TRAIN = os.environ.get("FEATURE_PATH_TRAIN", FEATURE_STORE_PATH)
FEATURE_PATH_VALID = os.environ.get("FEATURE_PATH_VALID", FEATURE_STORE_PATH)
_HELP["FEATURE_PATH_TRAIN"] = _HELP["FEATURE_PATH_VALID"] = (
    "Stage-1.5 feature cache dir (defaults to FEATURE_STORE_PATH)"
)

# ============================================================================
# Bundled in the repo (assets/) -- defaults work, override only if needed
# ============================================================================
CTRG_MODEL_PATH = os.environ.get("CTRG_MODEL_PATH", os.path.join(_ASSETS, "ctrg_text_config"))
CTRG_ANNOTATION_JSON = os.environ.get("CTRG_ANNOTATION_JSON", os.path.join(_ASSETS, "annotation.json"))
TEXT_LATENT_NPZ = os.environ.get(
    "TEXT_LATENT_NPZ",
    os.path.join(_ASSETS, "text_latent_feature_250812_bert_ipmiversion_valid_Qwen_extract_abnorm.npz"),
)
MED_CONFIG_BLIP = os.environ.get(
    "MED_CONFIG_BLIP", os.path.join(_REPO_ROOT, "ctrg", "modules", "configs", "med_config_blip.json")
)
BLEU_METRIC_PATH = os.environ.get("BLEU_METRIC_PATH", os.path.join(_ASSETS, "bleu", "bleu.py"))
# Only used by patch_selection.get_retrieval(), which has no callers.
ORGAN_CENTERS_NPY = os.environ.get(
    "ORGAN_CENTERS_NPY", os.path.join(_ASSETS, "text_embedding_organ_centers_train.npy")
)
REGION_REPORT_CSV = _user(
    "REGION_REPORT_CSV",
    "train_region_report.csv; only for the unused patch_selection.get_retrieval()",
)

# ============================================================================
# Data bundle ($CTRG_DATA_ROOT, released as ctrg_data.tar.gz; default <repo>/data)
#
#   $CTRG_DATA_ROOT/
#     ctrate/
#       reports/            train_reports.csv, dataset_radiology_text_reports_validation_reports.csv
#       labels/             dataset_multi_abnormality_labels_{train,valid}_predicted_labels.csv
#       region_report/      train_region_report_abnormality_exacted_shorted.csv,
#                           dataset_radgenome_files_validation_region_report_abnormalities_extracted_shorted0_60000.csv
#       metadata/           train_metadata.csv, dataset_metadata_validation_metadata.csv
#       report_hierarchy_tree/{train,valid}/<case>.json
#       mask_fast/{train,valid}/<case>.npy
#     text_embedding/ipmiversion/<case>_text_embedding.npy
# ============================================================================
CTRG_DATA_ROOT = os.environ.get("CTRG_DATA_ROOT", os.path.join(_REPO_ROOT, "data"))
_CTRATE_DATA = os.path.join(CTRG_DATA_ROOT, "ctrate")

CTRATE_TRAIN_REPORTS = os.environ.get("CTRATE_TRAIN_REPORTS", os.path.join(_CTRATE_DATA, "reports", "train_reports.csv"))
CTRATE_VALID_REPORTS = os.environ.get("CTRATE_VALID_REPORTS", os.path.join(_CTRATE_DATA, "reports", "dataset_radiology_text_reports_validation_reports.csv"))
CTRATE_TRAIN_LABELS = os.environ.get("CTRATE_TRAIN_LABELS", os.path.join(_CTRATE_DATA, "labels", "dataset_multi_abnormality_labels_train_predicted_labels.csv"))
CTRATE_VALID_LABELS = os.environ.get("CTRATE_VALID_LABELS", os.path.join(_CTRATE_DATA, "labels", "dataset_multi_abnormality_labels_valid_predicted_labels.csv"))
CTRATE_TRAIN_REGION_REPORT = os.environ.get("CTRATE_TRAIN_REGION_REPORT", os.path.join(_CTRATE_DATA, "region_report", "train_region_report_abnormality_exacted_shorted.csv"))
CTRATE_VALID_REGION_REPORT = os.environ.get("CTRATE_VALID_REGION_REPORT", os.path.join(_CTRATE_DATA, "region_report", "dataset_radgenome_files_validation_region_report_abnormalities_extracted_shorted0_60000.csv"))
CTRATE_TRAIN_METADATA = os.environ.get("CTRATE_TRAIN_METADATA", os.path.join(_CTRATE_DATA, "metadata", "train_metadata.csv"))
CTRATE_VALID_METADATA = os.environ.get("CTRATE_VALID_METADATA", os.path.join(_CTRATE_DATA, "metadata", "dataset_metadata_validation_metadata.csv"))
JSON_PATH_TRAIN = os.environ.get("JSON_PATH_TRAIN", os.path.join(_CTRATE_DATA, "report_hierarchy_tree", "train"))
JSON_PATH_VALID = os.environ.get("JSON_PATH_VALID", os.path.join(_CTRATE_DATA, "report_hierarchy_tree", "valid"))
CTRATE_TRAIN_MASK_FAST = os.environ.get("CTRATE_TRAIN_MASK_FAST", os.path.join(_CTRATE_DATA, "mask_fast", "train"))
CTRATE_VALID_MASK_FAST = os.environ.get("CTRATE_VALID_MASK_FAST", os.path.join(_CTRATE_DATA, "mask_fast", "valid"))
TEXT_EMBEDDING_DIR = os.environ.get("TEXT_EMBEDDING_DIR", os.path.join(CTRG_DATA_ROOT, "text_embedding", "ipmiversion"))

# ============================================================================
# GRPO / evaluation servers (HTTP). Change only if servers run elsewhere.
# ============================================================================
# Reference-model server (reference per-token logps for the KL term)
GRPO_REF_SERVER_URL = os.environ.get("GRPO_REF_SERVER_URL", "http://localhost:59888")
GRPO_REF_SERVER_PORT = int(os.environ.get("GRPO_REF_SERVER_PORT", "59888"))
# Reward generation + aggregation server (D_full writes the report, then it is scored)
GRPO_REWARD_SERVER_URL = os.environ.get("GRPO_REWARD_SERVER_URL", "http://localhost:60010")
GRPO_REWARD_SERVER_PORT = int(os.environ.get("GRPO_REWARD_SERVER_PORT", "60010"))
# F1 server (RadBERT classifier): GRPO F1 reward AND Stage-3 validation/test CE-F1
GRPO_F1_SERVER_URL = os.environ.get("GRPO_F1_SERVER_URL", "http://localhost:61000")
GRPO_F1_SERVER_PORT = int(os.environ.get("GRPO_F1_SERVER_PORT", "61000"))
# GREEN reward server
GRPO_GREEN_SERVER_URL = os.environ.get("GRPO_GREEN_SERVER_URL", "http://localhost:59879")
GRPO_GREEN_SERVER_PORT = int(os.environ.get("GRPO_GREEN_SERVER_PORT", "59879"))
