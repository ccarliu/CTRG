import copy
import os
import resource

import pytorch_lightning as pl
import torch
from CTRG.config import ex
from CTRG.datamodules.pretraining_medicat_datamodule import MedicatDataModule_3D_RATE_hr
from CTRG.modules import M3AETransformerSS_3D_lmae_rg_pretrain_v29

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])


    dm = MedicatDataModule_3D_RATE_hr(_config)

    # Module
    _config["image_size"] = [480, 480]
    _config["image_depth"] = 240
    _config["batch_size"] = _config["per_gpu_batchsize"]

    model = M3AETransformerSS_3D_lmae_rg_pretrain_v29(_config)

    # freeze the text encoder
    for name, param in model.named_parameters():
            
        if "text_model" in name or "text_decoder_ref" in name:
            param.requires_grad = False

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    loggers = [tb_logger]

    # Callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1) 
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = _config["max_epoch"] if max_steps is None else 800
    print(grad_steps, max_epochs)
    max_epochs = 20
    grad_steps = grad_steps // grad_steps
    # Trainer

    trainer = pl.Trainer(
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        strategy='ddp_find_unused_parameters_true',
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=30,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path =_config["resume_from"] )
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best", datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
