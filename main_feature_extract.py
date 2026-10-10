import copy
import os
import sys
import resource

# Make the repo root importable (ct_clip / ctvit / ctrg / paths_config).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist

from ctrg.config import ex
from ctrg.datamodules.pretraining_medicat_datamodule import MedicatDataModule_3D_RATE_hr
from ctrg.modules import M3AETransformerSS_3D_lmae_rg_pretrain_v29_Qwen_perorgan_mask
from paths_config import PRETRAIN_CKPT, FEATURE_STORE_PATH, require

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@ex.automain
def main(_config):
    """Stage 1.5: visual feature extraction / caching.

    Runs the Stage-1 pretrained CT-ViT perorgan encoder (`.infer`) over every CT
    volume and caches the per-case outputs to `FEATURE_STORE_PATH`:
        {name}selected_patch.npy  - top-K local patch tokens per structure
        {name}image_feature.npy   - full image feature pool
        {name}local_text.npy      - structure-wise text embedding
        {name}local_image.npy     - structure-wise visual embedding
    These are later consumed by CTRGDataset_RATE_hr_sim_fea (feature_path) in the
    feature-selection and decoder-finetuning stages.

    Multi-GPU: launch with torchrun; samples are statically sharded across ranks
    (idx % world_size == rank). Already-cached cases are skipped (resumable).
    """
    _config = copy.deepcopy(_config)
    _split = os.environ.get("EXTRACT_SPLIT", "train")
    require("BIOMEDVLP_BERT", "PRETRAIN_CKPT", "FEATURE_STORE_PATH",
            "CTRATE_TRAIN_IMAGES" if _split == "train" else "CTRATE_VALID_IMAGES")
    pl.seed_everything(_config["seed"] + int(os.environ.get("RANK", "0")))

    distributed, rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0

    _config["num_workers"] = max(2, _config.get("num_workers", 8))
    _config["image_size"] = [480, 480]
    _config["image_depth"] = 240
    _config["per_gpu_batchsize"] = 1
    _config["mask_rate"] = 0
    _config["hidden_size"] = 768

    # Raw-image datamodule (feeds real volumes into the encoder).
    dm = MedicatDataModule_3D_RATE_hr(_config)
    dm.setup(None)

    # Which split to extract: train / val / test (default: train).
    split = os.environ.get("EXTRACT_SPLIT", "train")
    if split == "train":
        dataloader = dm.train_dataloader()
    elif split == "val":
        dataloader = dm.val_dataloader()
    else:
        dataloader = dm.test_dataloader()

    store_path = FEATURE_STORE_PATH
    if is_main:
        os.makedirs(store_path, exist_ok=True)
    if distributed:
        dist.barrier()

    # Build the Stage-1 pretrained encoder and load its checkpoint.
    model = M3AETransformerSS_3D_lmae_rg_pretrain_v29_Qwen_perorgan_mask(_config)
    if not os.path.exists(PRETRAIN_CKPT):
        raise FileNotFoundError(f"PRETRAIN_CKPT not found: '{PRETRAIN_CKPT}' (Stage-1 encoder checkpoint)")
    ck = torch.load(PRETRAIN_CKPT, map_location="cpu")["state_dict"]
    model.load_state_dict(ck, strict=False)
    if is_main:
        print(f"[feature_extract] loaded encoder checkpoint: {PRETRAIN_CKPT}")
    model = model.to(device)
    model.eval()

    def _save(name, ret):
        np.save(os.path.join(store_path, name + "selected_patch.npy"), ret["selected_patch"].detach().cpu().numpy())
        np.save(os.path.join(store_path, name + "image_feature.npy"), ret["image_feature"].detach().cpu().numpy())
        np.save(os.path.join(store_path, name + "local_text.npy"), ret["local_text"].detach().cpu().numpy())
        np.save(os.path.join(store_path, name + "local_image.npy"), ret["local_image"].detach().cpu().numpy())

    for idx, batch in enumerate(dataloader):
        # Static sharding: each rank handles disjoint samples.
        if idx % world_size != rank:
            continue

        patient_name = batch["data_name"][0]
        if os.path.exists(os.path.join(store_path, patient_name + "selected_patch.npy")):
            continue

        batch_n = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        # The raw `_hr_sim` dataset does not emit a token-level structural `mask`.
        # `mask` is only used inside infer to compute the (discarded) sparse loss,
        # so inject a zero mask sized to the encoder's actual token count.
        if "mask" not in batch_n:
            try:
                with torch.no_grad():
                    feats = model.vision_encoder(batch_n["image"].unsqueeze(1), return_encoded_tokens=True)
                _, x1, x2, x3, _ = feats.shape
                n_tokens = x1 * x2 * x3
                batch_n["mask"] = torch.zeros((batch_n["image"].shape[0], 10, n_tokens), device=device)
            except Exception as e:
                print(f"[rank {rank}] WARN could not build dummy mask for {patient_name}: {e}")

        try:
            with torch.no_grad():
                ret = model.infer(batch_n)
            _save(patient_name, ret)
            if is_main:
                print(f"  saved {patient_name}: selected_patch={tuple(ret['selected_patch'].shape)}")
        except Exception as e:
            print(f"[rank {rank}] ERROR processing {patient_name}: {e}")
            continue

        if idx % 20 == 0 and is_main:
            print(f"[rank {rank}/{world_size}] processed idx={idx}, name={patient_name}")

    cleanup_distributed(distributed)
