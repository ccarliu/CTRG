import copy
import os
import sys
import resource
import warnings

# Make the repo root importable (ct_clip / ctvit / ctrg / GRPO / classifier / paths_config).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
import torch

from ctrg.config import ex
from ctrg.datamodules.pretraining_medicat_datamodule import MedicatDataModule_3D_RATE_hr_fea
from ctrg.modules.report_generation_v38_llm_v3_GRPO import (
    M3AETransformerSS_3D_lmae_rg_v38_llm_v3_GRPO,
)
from paths_config import GRPO_INIT_CKPT, require

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

warnings.filterwarnings("ignore", category=UserWarning)


@ex.automain
def main(_config):
    """Stage 4: GRPO collaborative refinement (client).

    Trains the abnormality-proposal policy D_pro via GRPO. The reference model,
    reward (F1 / GREEN) models, and the report-generation D_full all run as
    separate HTTP servers (see GRPO/). This client only holds the trainable
    policy and computes the GRPO loss.

    Prerequisite: start the servers first (GRPO/run_llama_ref.sh,
    GRPO/run_llama_ref_think.sh, and one of GRPO/run_F1_reward.sh /
    GRPO/run_green_reward.sh).
    """
    _config = copy.deepcopy(_config)
    require("BIOMEDVLP_BERT", "LLM_DECODER_PATH", "FEATURE_PATH_TRAIN", "FEATURE_PATH_VALID",
            "GRPO_INIT_CKPT")
    pl.seed_everything(_config["seed"])

    # == Data module (cached-feature pipeline) ==
    dm = MedicatDataModule_3D_RATE_hr_fea(_config)
    dm.eval_batch_size = 4
    _config["image_size"] = [480, 480]
    _config["image_depth"] = 240
    _config["batch_size"] = 1
    _config["per_gpu_batchsize"] = 1
    _config["selected_patch"] = 9

    # == Model (GRPO policy client) ==
    model = M3AETransformerSS_3D_lmae_rg_v38_llm_v3_GRPO(_config)

    # == Warm-start from the Stage-3 finetuned checkpoint ==
    if GRPO_INIT_CKPT and os.path.exists(GRPO_INIT_CKPT):
        ck = torch.load(GRPO_INIT_CKPT, map_location="cpu")["state_dict"]
        model.load_state_dict(ck, strict=False)
        print(f"[grpo] loaded init checkpoint: {GRPO_INIT_CKPT}")
    else:
        print(f"[grpo] no init checkpoint (GRPO_INIT_CKPT='{GRPO_INIT_CKPT}')")
    # Generated reports (validation) are appended to <ckpath>*.csv.
    pred_dir = os.path.join(_config["log_dir"], "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    model.ckpath = os.path.join(pred_dir, "val_grpo")

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
        save_top_k=5,
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
    _config["val_check_interval"] = 600

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
