import copy
import os
import sys
import resource
import warnings

# Make the repo root importable (ct_clip / ctvit / ctrg / paths_config).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
import torch

from ctrg.config import ex
from ctrg.datamodules.pretraining_medicat_datamodule import MedicatDataModule_3D_RATE_hr_fea
from ctrg.modules.report_generation_v38_llm_v3 import M3AETransformerSS_3D_lmae_rg_v38_llm_v3
from paths_config import FINETUNE_INIT_CKPT, SELECTOR_INIT_CKPT, require

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

warnings.filterwarnings("ignore", category=UserWarning)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    require("BIOMEDVLP_BERT", "LLM_DECODER_PATH", "FEATURE_PATH_TRAIN", "FEATURE_PATH_VALID")
    if _config["test_only"]:
        require("FINETUNE_INIT_CKPT")
    pl.seed_everything(_config["seed"])

    _config["selected_patch"] = 9
    _config["max_steps"] = 400000
    _config["learning_rate"] = 1e-4

    # == Model ==
    model = M3AETransformerSS_3D_lmae_rg_v38_llm_v3(_config)

    # == Data module ==
    dm = MedicatDataModule_3D_RATE_hr_fea(_config)
    _config["image_size"] = [480, 480]
    _config["image_depth"] = 240
    _config["batch_size"] = 1
    _config["per_gpu_batchsize"] = 1

    # == Warm-start from a (pretrained) checkpoint ==
    if FINETUNE_INIT_CKPT and os.path.exists(FINETUNE_INIT_CKPT):
        ck = torch.load(FINETUNE_INIT_CKPT, map_location="cpu")["state_dict"]
        model.load_state_dict(ck, strict=False)
        print(f"[finetune] loaded init checkpoint: {FINETUNE_INIT_CKPT}")
    else:
        print(f"[finetune] no init checkpoint (FINETUNE_INIT_CKPT='{FINETUNE_INIT_CKPT}')")

    # == Load the Stage-2 AP-LTF selector (used by D_full, train_mode=full) ==
    # When evaluating (test_only=True) FINETUNE_INIT_CKPT is the full Stage-3
    # checkpoint, which already contains the selector.
    if not _config["test_only"] and SELECTOR_INIT_CKPT:
        if not os.path.exists(SELECTOR_INIT_CKPT):
            raise FileNotFoundError(f"SELECTOR_INIT_CKPT not found: '{SELECTOR_INIT_CKPT}'")
        ck = torch.load(SELECTOR_INIT_CKPT, map_location="cpu")["state_dict"]
        nck = {k[len("selector."):]: v for k, v in ck.items() if k.startswith("selector.")}
        if not nck:
            raise KeyError(f"no 'selector.*' weights in SELECTOR_INIT_CKPT='{SELECTOR_INIT_CKPT}'")
        model.selector.load_state_dict(nck)
        print(f"[finetune] loaded AP-LTF selector ({len(nck)} tensors) from: {SELECTOR_INIT_CKPT}")
    elif _config["train_mode"] == "full" and not _config["test_only"]:
        print("[finetune] WARNING: SELECTOR_INIT_CKPT is empty; D_full will use a randomly initialised selector")

    # Generated reports (validation / test) are appended to <ckpath>.csv.
    pred_dir = os.path.join(_config["log_dir"], "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    if _config["test_only"]:
        tag = os.path.splitext(os.path.basename(FINETUNE_INIT_CKPT or "no_ckpt"))[0]
        model.ckpath = os.path.join(pred_dir, f"test_{_config['train_mode']}_{tag}")
    else:
        model.ckpath = os.path.join(pred_dir, f"val_{_config['train_mode']}")

    # == Freeze the vision encoder ==
    for name, param in model.named_parameters():
        if "vision_encoder" in name:
            param.requires_grad = False

    # == Loggers ==
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    loggers = [tb_logger]

    # == Callbacks ==
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
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
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = 20
    grad_steps = 1

    # == Trainer ==
    trainer = pl.Trainer(
        devices=num_gpus,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=False,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"],
        strategy="ddp_find_unused_parameters_true",
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best", datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
