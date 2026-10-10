# CT-RATE Data Preprocessing

Preprocessing scripts for CT scans, integrated into this standalone repo. They
convert raw `*.nii.gz` volumes into the `*.npz` files consumed by the dataset
(`ctrg/datasets/pretraining_ctrg_dataset.py`). Steps: normalize to Hounsfield
units (HU), clip HU to `[-1000, 1000]`, adjust orientation, and resample x/y/z
spacing to `0.75 / 0.75 / 1.5`.

Unlike the original scripts (which hardcoded absolute paths and had to be copied
into each data subfolder), all I/O paths here are read from the repo-level
`paths_config.py` and can be overridden with environment variables. You can run
the scripts in place from the repo root.

## Scripts

| Script | Input (raw) | Output | Purpose |
|---|---|---|---|
| `preprocess_ctrate_train.py` | `CTRATE_RAW_TRAIN_DIR` + `CTRATE_TRAIN_METADATA` | `CTRATE_TRAIN_IMAGES` (`*.npz`) | preprocess training volumes |
| `preprocess_ctrate_valid.py` | `CTRATE_RAW_VALID_DIR` + `CTRATE_VALID_METADATA` | `CTRATE_VALID_IMAGES` (`*.npz`) | preprocess validation volumes |
| `preprocess_ctrate_mask.py` | `CTRATE_RAW_SEG_DIR` + `CTRATE_TRAIN_METADATA` | `CTRATE_MASK_PREPROCESSED` (`*.npz`) | preprocess structural segmentation masks (nearest-neighbor resample) |
| `datacheck.py` | `CTRATE_TRAIN_IMAGES` | — | scan output `*.npz`, delete unreadable/corrupt files |

> `preprocess_ctrate_train.py` also contains an optional `process_file_pillar`
> variant (raw-HU, isotropic 1.25 mm, 256³ `.npy`) for a Pillar-0/Atlas encoder;
> it is **not** used by the default `__main__` and is kept only for reference.

## Relevant environment variables (see repo-level `paths_config.py`)

| Env var | Default | Meaning |
|---|---|---|
| `CTRATE_RAW_TRAIN_DIR` | `.../data_volumes/dataset/train` | raw training nii.gz dir |
| `CTRATE_RAW_VALID_DIR` | `.../data_volumes/dataset/valid` | raw validation nii.gz dir |
| `CTRATE_RAW_SEG_DIR` | `.../ReXGroundingCT/segmentations` | raw segmentation nii.gz dir |
| `CTRATE_TRAIN_METADATA` | `$CTRG_DATA_ROOT/ctrate/metadata/train_metadata.csv` | per-volume spacing / rescale |
| `CTRATE_VALID_METADATA` | `$CTRG_DATA_ROOT/ctrate/metadata/dataset_metadata_validation_metadata.csv` | per-volume spacing / rescale |
| `CTRATE_TRAIN_IMAGES` | `.../train_preprocessed` | **output** of train preprocessing (dataset input) |
| `CTRATE_VALID_IMAGES` | `.../valid_preprocessed` | **output** of valid preprocessing (dataset input) |
| `CTRATE_MASK_PREPROCESSED` | `.../mask_preprocessed` | **output** of mask preprocessing |
| `PREPROCESS_NUM_WORKERS` | `10` | multiprocessing pool size |

The metadata CSV must contain the columns `VolumeName`, `RescaleIntercept`,
`XYSpacing`, `ZSpacing` for every volume.

## Usage

```bash
cd /path/to/ctrg_pretrain

# (optional) point paths at your own copies
export CTRATE_RAW_TRAIN_DIR=/data/CTRATE/train
export CTRATE_TRAIN_METADATA=/data/CTRATE/train_metadata.csv
export CTRATE_TRAIN_IMAGES=/data/CTRATE/train_preprocessed
export PREPROCESS_NUM_WORKERS=16

# 1) preprocess volumes
python data_preprocess/preprocess_ctrate_train.py
python data_preprocess/preprocess_ctrate_valid.py

# 2) (optional) preprocess structural masks
python data_preprocess/preprocess_ctrate_mask.py

# 3) (optional) verify outputs and drop corrupt npz
python data_preprocess/datacheck.py
```

After preprocessing, the `CTRATE_{TRAIN,VALID}_IMAGES` directories feed directly
into Stage-1 pretraining (`main_3D.py`) and the Stage-1.5 feature extraction.

> `datacheck.py` deletes unreadable files by default
> (`delete_on_error=True`). Run it with `delete_on_error=False` first if you want
> a dry run.

## Notes on mask output path

You normally do **not** need to run the mask script: the training code only reads
the fast structural masks (`CTRATE_{TRAIN,VALID}_MASK_FAST`, one `<case>.npy` per
volume), which ship with the data bundle under `$CTRG_DATA_ROOT/ctrate/mask_fast/`
and cover every case. A missing fast mask raises a `FileNotFoundError`.

`preprocess_ctrate_mask.py` writes resampled `*.npz` masks to
`CTRATE_MASK_PREPROCESSED`; these are an intermediate format and are not read by
the dataset directly.
