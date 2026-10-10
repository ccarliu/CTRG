import copy
import os
import sys
import resource

# Make the repo root importable so that top-level packages
# (ct_clip, ctvit, ctrg, paths_config) resolve regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
import torch

from ctrg.config import ex
from ctrg.datamodules.pretraining_medicat_datamodule import MedicatDataModule_3D_RATE_hr
from ctrg.modules import M3AETransformerSS_3D_lmae_rg_pretrain_v29_Qwen_perorgan_mask
from paths_config import PRETRAIN_CKPT, require

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    require("BIOMEDVLP_BERT", "CTRATE_TRAIN_IMAGES", "CTRATE_VALID_IMAGES")
    pl.seed_everything(_config["seed"])

    # == Data module ==
    # Stage 1 trains the CT-ViT encoder on raw preprocessed CT volumes, so it needs
    # CTRGDataset_RATE_hr_sim (raw image + fast mask + per-case text embedding).
    # The *_hr_fea variant reads cached features and is for Stages 2-4 only.
    dm = MedicatDataModule_3D_RATE_hr(_config)

    _config["image_size"] = [480, 480]
    _config["image_depth"] = 240
    _config["batch_size"] = _config["per_gpu_batchsize"]
    _config["mask_rate"] = 0.25
    _config["hidden_size"] = 768

    # == Model ==
    model = M3AETransformerSS_3D_lmae_rg_pretrain_v29_Qwen_perorgan_mask(_config)

    # == (Optional) warm-start from a pretrain checkpoint ==
    if PRETRAIN_CKPT and os.path.exists(PRETRAIN_CKPT):
        ck = torch.load(PRETRAIN_CKPT, map_location="cpu")["state_dict"]
        model.load_state_dict(ck, strict=False)
        print(f"[main_3D] loaded checkpoint: {PRETRAIN_CKPT}")
    else:
        print(f"[main_3D] training from scratch (PRETRAIN_CKPT='{PRETRAIN_CKPT}')")

    # == Freeze frozen text branches ==
    for name, param in model.named_parameters():
        if "text_model" in name or "text_decoder_ref" in name:
            param.requires_grad = False

    # == Loggers ==
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    loggers = [tb_logger]

    # == Callbacks ==
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # == Training hyper-parameters ==
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = 40
    grad_steps = 1

    # == Trainer ==
    trainer = pl.Trainer(
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        strategy="ddp_find_unused_parameters_true",
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=30,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
    else:
        trainer.test(model, datamodule=dm)
