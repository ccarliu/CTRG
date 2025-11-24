import copy
import os
import resource

import pytorch_lightning as pl

from m3ae.config import ex
from m3ae.datamodules.multitask_datamodule import MTDataModule
from m3ae.datamodules.pretraining_medicat_datamodule import MedicatDataModule_3D, MedicatDataModule_3D_RATE, MedicatDataModule_3D_RATE_hr
from m3ae.datasets.pretraining_ctrg_dataset import CTRGDataset
from m3ae.modules import M3AETransformerSS_3D_lmae_rg_pretrain_v29

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaConfig, RobertaModel, BertTokenizer, BertModel

import numpy as np

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# import umap
from sklearn.preprocessing import StandardScaler
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Data modules

    _config["num_workers"] = 8
    dm = MedicatDataModule_3D_RATE_hr(_config)

    # Module
    _config["image_size"] = [480, 480]
    _config["image_depth"] = 240
    _config["batch_size"] = 1

    _config["mask_rate"] = 0

    dm.setup("test")

    model = M3AETransformerSS_3D_lmae_rg_pretrain_v29(_config)

    
    load_path = load path
    store_path = store path

    if not os.path.exists(store_path):
        os.mkdir(store_path)

    ckpt = torch.load(load_path, map_location="cpu")

    model.load_state_dict(ckpt["state_dict"])

    model = model.cuda()

    dm.setup("test")



    for idx, batch in enumerate(dm.test_dataloader()):
        
        patient_name = batch["data_name"][0]
        
        print(idx, patient_name, end = "\r")
        

        names.append(patient_name)

        batch_n = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
       
        with torch.no_grad():  
            # model.eval()  
            ret = model.infer(batch_n, early_quit = True)
        
        
        np.save(os.path.join(store_path, patient_name + "selected_patch" + ".npy"), ret["selected_patch"].cpu().numpy()) # Top 10-feature
        np.save(os.path.join(store_path, patient_name + "image_feature" + ".npy"), ret["image_feature"].cpu().numpy()) # all image feature
        np.save(os.path.join(store_path, patient_name + "local_text" + ".npy"), ret["local_text"].cpu().numpy()) # text embedding
        np.save(os.path.join(store_path, patient_name + "local_image" + ".npy"), ret["local_image"].cpu().numpy()) # visual structural embedding

    for idx, batch in enumerate(dm.train_dataloader()):
        
        patient_name = batch["data_name"][0]
        
        print(idx, patient_name, end = "\r")
        

        names.append(patient_name)

        batch_n = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
       
        with torch.no_grad():  
            # model.eval()  
            ret = model.infer(batch_n, early_quit = True)
        
        
        np.save(os.path.join(store_path, patient_name + "selected_patch" + ".npy"), ret["selected_patch"].cpu().numpy()) # Top 10-feature
        np.save(os.path.join(store_path, patient_name + "image_feature" + ".npy"), ret["image_feature"].cpu().numpy()) # all image feature
        np.save(os.path.join(store_path, patient_name + "local_text" + ".npy"), ret["local_text"].cpu().numpy()) # text embedding
        np.save(os.path.join(store_path, patient_name + "local_image" + ".npy"), ret["local_image"].cpu().numpy()) # visual structural embedding

