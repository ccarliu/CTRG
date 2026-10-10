import functools

import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler

from .gloria_loss.gloria_loss import global_loss, local_loss

from .dist_utils import all_gather
from scipy.special import expit
from sklearn.metrics import classification_report

import warnings

import requests

from .online_utils import get_results
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)
import csv
from pathlib import Path

import numpy as np


def compute_localcl(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    # print(infer['global_text'].shape, infer['global_image'].shape)
    #print(infer['local_text'].shape, infer['local_image'].shape)

    # print(infer['global_image'].squeeze().requires_grad)
    loss_global, loss_global2 = global_loss(infer['global_image'].squeeze(), infer['global_text'].squeeze())
    loss_local, loss_local2, attn = local_loss(infer['local_image'].squeeze(0).permute(0,2,1), infer['local_text'].squeeze(0).permute(0,2,1), infer['length'])

    # print(loss_global, loss_local, loss_global2, loss_local2, attn[0].max(), attn[0].min())
    ret = {
        "mlm_loss": loss_global+loss_global2, #+loss_local2 +loss_local,
        "infer": infer,
    }

    # print(infer["captions"])
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = -loss
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", -loss)
    
    return ret

def compute_rg_ce_CTRG(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    
    loss_lm = infer["decoder_output"].loss
    
    #loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    #print(loss_cls, "xxxxxxxxxxxxx")

    ret = {
        "rg_loss": loss_lm,
        "infer": infer,
    }
    
    phase = "train" if pl_module.training else "val"
    # print(getattr(pl_module, f"{phase}_rg_loss").total)
    # print(infer["captions"])
    # print(infer["text_ori"])

    if not pl_module.training:
        store_path = getattr(pl_module, f"ckpath") + ".txt"
        text_file = open(store_path, "a+")
        text_file.writelines("****" + "\n")
        text_file.writelines(infer["captions"][0] + "\n")
        # print(infer["captions"][0])
    
    loss = getattr(pl_module, f"{phase}_rg_loss")(ret["rg_loss"])
    # print(phase ,1)
    #BLEU_1, _, _, ROUGE_L = getattr(pl_module, f"{phase}_rg_BLEU_1")({1:infer["text_ori"][:1]}, {1:infer["captions"][:1]})
    # print(phase ,2)
    # print(BLEU_1, ROUGE_L, loss)
    
    #####################################
    # print(infer["captions"])\
    cf1s = []
    for l in range(len(infer["captions"])):
        #try:
        try:
            tokenized_imp = pl_module.tokenizer_ce.tokenize(infer["captions"][l])
            res = pl_module.tokenizer_ce.encode_plus(tokenized_imp)['input_ids']
        except:
            tokenized_imp = pl_module.tokenizer_ce.tokenize("None. this is a take place sentence.")
            res = pl_module.tokenizer_ce.encode_plus(tokenized_imp)['input_ids']
        
        if len(res) > 512: #length exceeds maximum size
            #print("report length bigger than 512")
            res = res[:511] + [pl_module.tokenizer_ce.sep_token_id]
        
        res = torch.LongTensor(res).unsqueeze(0).to(infer["images"].device)
        # print(res.shape)
        masks = torch.ones(res.size(0), res.size(1), dtype=torch.float).to(infer["images"].device)

        with torch.no_grad():
            out = pl_module.classificer(res, masks)
        
        pred_labels = np.array([out[l].argmax(dim=1).cpu().numpy() for l in range(len(out))])==1 #[pred_labels<0.5]=0

        #print(pred_labels, infer["obver_label"])

        obver_label = infer["obver_label"][l].cpu().numpy()
        
        #print(obver_label, pred_labels)
        # print(classification_report(obver_label.flatten(), pred_labels.flatten(), output_dict=True))
        cf1 = classification_report(obver_label.flatten(), pred_labels.flatten(), output_dict=True, zero_division=0).get('1', {}).get('f1-score', 0.0)
        cf1s.append(cf1)
        #except:
            #cf1 = 0.

    cf1 = np.mean(np.array(cf1s))
    #print(cf1)
    #########################################

    pl_module.log(f"rg/{phase}/loss", loss)
    pl_module.log(f"rg/{phase}/accuracy", getattr(pl_module, f"{phase}_rg_BLEU_1")(cf1))
    
    return ret


def compute_rg_dpo(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    
    loss_lm = infer["dpo_loss"] # + infer["decoder_output"].loss
    
    #loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    #print(loss_cls, "xxxxxxxxxxxxx")

    ret = {
        "rg_loss": loss_lm,
        "infer": infer,
    }
    
    phase = "train" if pl_module.training else "val"
    # print(getattr(pl_module, f"{phase}_rg_loss").total)
    # print(infer["captions"])
    # print(infer["text_ori"])

    if not pl_module.training:
        store_path = getattr(pl_module, f"ckpath") + ".txt"
        text_file = open(store_path, "a+")
        text_file.writelines("****" + "\n")
        text_file.writelines(infer["captions"][0] + "\n")
    
    loss = getattr(pl_module, f"{phase}_rg_loss")(ret["rg_loss"])
    # print(phase ,1)
    #BLEU_1, _, _, ROUGE_L = getattr(pl_module, f"{phase}_rg_BLEU_1")({1:infer["text_ori"][:1]}, {1:infer["captions"][:1]})
    # print(phase ,2)
    # print(BLEU_1, ROUGE_L, loss)
    
    #####################################
    encodings = pl_module.tokenizer_ce(infer["captions"], return_tensors='pt',max_length=512,padding='max_length',truncation=True)
        
    input_ids = encodings['input_ids'].to(infer["text_ids"].device)
    attention_mask = encodings['attention_mask'].to(infer["text_ids"].device)

    with torch.no_grad():
        pred_logits = pl_module.classificer(input_ids,attention_mask).cpu().numpy()

    # print(pred_logits.shape)
    # pred_logits = pred_logits[1:]
    pred_labels = expit(pred_logits)
    
    pred_labels[pred_labels>=0.5]=1
    pred_labels[pred_labels<0.5]=0

    obver_label = infer["obver_label"].cpu().numpy()

    # print(classification_report(obver_label.flatten(), pred_labels.flatten(), output_dict=True))
    cf1 = classification_report(obver_label.flatten(), pred_labels.flatten(), output_dict=True).get('1', {}).get('f1-score', 0.0)

    # print(cf1)
    #########################################

    pl_module.log(f"rg/{phase}/loss", loss)
    pl_module.log(f"rg/{phase}/accuracy", getattr(pl_module, f"{phase}_rg_BLEU_1")(cf1))
    
    return ret

def compute_rg(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    
    loss_lm = infer["decoder_output"].loss
    
    #loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    #print(loss_cls, "xxxxxxxxxxxxx")

    ret = {
        "rg_loss": loss_lm,
        "infer": infer,
    }
    
    phase = "train" if pl_module.training else "val"
    # print(getattr(pl_module, f"{phase}_rg_loss").total)
    # print(infer["captions"])
    # print(infer["text_ori"])

    if not pl_module.training:
        store_path = getattr(pl_module, f"ckpath") + ".txt"
        text_file = open(store_path, "a+")
        text_file.writelines("****" + "\n")
        text_file.writelines(infer["captions"][0] + "\n")
        # print(infer["captions"][0])
    
    loss = getattr(pl_module, f"{phase}_rg_loss")(ret["rg_loss"])
    # print(phase ,1)
    #BLEU_1, _, _, ROUGE_L = getattr(pl_module, f"{phase}_rg_BLEU_1")({1:infer["text_ori"][:1]}, {1:infer["captions"][:1]})
    # print(phase ,2)
    # print(BLEU_1, ROUGE_L, loss)
    
    #####################################
    if not pl_module.training:
        encodings = pl_module.tokenizer_ce(infer["captions"], return_tensors='pt',max_length=512,padding='max_length',truncation=True)
            
        input_ids = encodings['input_ids'].to(infer["images"].device)
        attention_mask = encodings['attention_mask'].to(infer["images"].device)

        with torch.no_grad():
            pred_logits = pl_module.classificer(input_ids,attention_mask).cpu().numpy()

        # print(pred_logits.shape)
        # pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)
        
        pred_labels[pred_labels>=0.5]=1
        pred_labels[pred_labels<0.5]=0

        obver_label = infer["obver_label"].cpu().numpy()

        # print(classification_report(obver_label.flatten(), pred_labels.flatten(), output_dict=True))
        # print(obver_label.shape)
        cf1 = []
        for ii in range(obver_label.shape[0]):
            cf1.append(classification_report(obver_label[ii].flatten(), pred_labels[ii].flatten(), output_dict=True).get('1', {}).get('f1-score', 0.0))
        cf1 = np.mean(cf1)
    else:
        cf1 = 0.0

    
    #########################################

    pl_module.log(f"rg/{phase}/loss", loss)
    pl_module.log(f"rg/{phase}/accuracy", getattr(pl_module, f"{phase}_rg_BLEU_1")(cf1))
    
    return ret


def compute_rg_F1_online(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    
    loss_lm = infer["decoder_output"].loss
    
    #loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    #print(loss_cls, "xxxxxxxxxxxxx")

    ret = {
        "rg_loss": loss_lm,
        "infer": infer,
    }
    
    phase = "train" if pl_module.training else "val"
    # print(getattr(pl_module, f"{phase}_rg_loss").total)
    # print(infer["captions"])
    # print(infer["text_ori"])

    if not pl_module.training:
        store_path = getattr(pl_module, f"ckpath") + ".csv"
        # 把 Tensor 整体搬到 numpy，避免循环里频繁 .cpu()
        obver_np = infer["obver_label"].cpu().numpy()

        # 安全写入
        with Path(store_path).open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 同时迭代三列，长度不一致时按最短的来
            for name, label, cap, text in zip(infer["data_name"], obver_np, infer["captions"], infer["text_ori"]):
                # print(name, label, cap)
                writer.writerow([name, label, cap, text])
        # text_file.writelines(infer["captions"][0] + "\n")
        # print(infer["captions"][0])
    
    loss = getattr(pl_module, f"{phase}_rg_loss")(ret["rg_loss"])
    
    if not pl_module.training:
        pred_abnormality = get_results(infer["captions"])

        obver_label = infer["obver_label"].cpu().numpy()

        # print(obver_label.shape)
        cf1 = []
        for ii in range(obver_label.shape[0]):
            cc = classification_report(obver_label[ii].flatten(), pred_abnormality[ii].flatten(), output_dict=True).get('1', {}).get('f1-score', -1)
            if cc == -1:
                if pred_abnormality.sum() == 0:
                    cf1.append(1.0)
                else:
                    cf1.append(0.0)
            else:
                cf1.append(cc)

        cf1 = np.mean(cf1)
    else:
        cf1 = 0
    # print(cf1)
    pl_module.log(f"rg/{phase}/loss", loss)
    pl_module.log(f"rg/{phase}/accuracy", getattr(pl_module, f"{phase}_rg_BLEU_1")(cf1))
    
    return ret

def compute_rg_F1_online_grpo(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    
    loss_lm = infer["grpo_loss"]
    
    #loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    #print(loss_cls, "xxxxxxxxxxxxx")

    ret = {
        "rg_loss": loss_lm,
        "infer": infer,
    }
    
    phase = "train" if pl_module.training else "val"
    # print(getattr(pl_module, f"{phase}_rg_loss").total)
    # print(infer["captions"])
    # print(infer["text_ori"])

    if not pl_module.training:
        store_path = getattr(pl_module, f"ckpath") + ".csv"
        # 把 Tensor 整体搬到 numpy，避免循环里频繁 .cpu()
        obver_np = infer["obver_label"].cpu().numpy()

        # 安全写入
        with Path(store_path).open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 同时迭代三列，长度不一致时按最短的来
            for name, label, cap in zip(infer["data_name"], obver_np, infer["captions"]):
                # print(name, label, cap)
                writer.writerow([name, label, cap])
        # print(infer["captions"][0])
    
    loss = getattr(pl_module, f"{phase}_rg_loss")(ret["rg_loss"])
    
    if not pl_module.training:
        pred_abnormality = get_results(infer["captions"])

        obver_label = infer["obver_label"].cpu().numpy()

        # print(obver_label.shape)
        cf1 = []
        for ii in range(obver_label.shape[0]):
            cc = classification_report(obver_label[ii].flatten(), pred_abnormality[ii].flatten(), output_dict=True).get('1', {}).get('f1-score', -1)
            if cc == -1:
                if pred_abnormality.sum() == 0:
                    cf1.append(1.0)
                else:
                    cf1.append(0.0)
            else:
                cf1.append(cc)

        cf1 = np.mean(cf1)
        
    else:
        cf1 = 0
    # print(phase ,2)
    # print(BLEU_1, ROUGE_L, loss)

    pl_module.log(f"rg/{phase}/loss", loss)
    pl_module.log(f"rg/{phase}/accuracy", getattr(pl_module, f"{phase}_rg_BLEU_1")(cf1))
    
    return ret


def info_nce(query, positive_key, negative_keys=None, mask = None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    # mask: just when negative_logits is None
    if mask is None:
        text_sim = query @ transpose(query)
        mask = torch.ones_like(logits, device = logits.device)
        mask[text_sim == 1] = 0 
        mask += torch.eye(logits.shape[-1], device = logits.device)

    # print(text_sim, mask, logits.shape) # 27 * 27

    return (F.cross_entropy(logits / temperature, labels, reduction=reduction) * mask) .mean()


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_head(infer["multi_modal_text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    loss_cl = info_nce(infer["local_text"].squeeze(0), infer["local_image"].squeeze(0))
    loss_cl += info_nce(infer["local_image"].squeeze(0), infer["local_text"].detach().squeeze(0))

    print(loss_cl, mlm_loss)


    ret = {
        "mlm_loss": mlm_loss + loss_cl,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    pl_module.log(f"mlm/{phase}/loss", loss + loss_cl)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    
    return ret

def compute_mlm2(pl_module, batch): # just for glovia
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
   # mlm_logits = pl_module.mlm_head(infer["multi_modal_text_feats"])
    #mlm_labels = infer["text_labels"]

    #mlm_loss = F.cross_entropy(
    #    mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
    #    mlm_labels.view(-1),
    #    ignore_index=-100,
    #)
    # loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    
    #sim_i2t = infer["local_image"] @ infer["local_text_m_all"] / 0.07
    # = infer["local_text"] @ infer["local_image_m_all"] / 0.07
                             
    #loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*infer["sim_i2t_targets"],dim=1).mean()
    #loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*infer["sim_t2i_targets"],dim=1).mean() 

    # loss_ita = (loss_i2t+loss_t2i)/2
    n,b,c,l=infer['local_image_word'].shape
    n,b,c2,l=infer['local_text_word'].shape

    # print(infer["local_text"].shape)

    loss_cl = info_nce(infer["local_text"].reshape(n*b, -1), infer["local_image"].reshape(n*b, -1))
    loss_cl += info_nce(infer["local_image"].reshape(n*b, -1), infer["local_text"].reshape(n*b, -1))
   #  print(infer['local_image_word'].shape, infer['local_image_word'].shape)

   #  print(infer['length'])
    loss_local, loss_local2, attn = local_loss(infer['local_image_word'].reshape(n*b, c, l).permute(0,2,1), infer['local_text_word'].reshape(n*b, c2, l).permute(0,2,1), torch.cat(infer['length'], 0))

    # print((loss_local + loss_local2) / 2) #, loss_cls)

    # exit(0)

    ret = {
        "mlm_loss": loss_local + loss_local2 + loss_cl / 2, # + loss_ita, # (loss_local + loss_local2) / 2 * 0.1 + , # +  , # + loss_cls * 0.3,
        #"mlm_logits": mlm_logits,
        #"mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    
    # print(loss_cl)
    # print(infer["local_text"].shape)
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = -1*(loss_local + loss_local2)
    pl_module.log(f"mlm/{phase}/loss", loss_local + loss_local2)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    
    return ret

def stable_sim_loss(sim_i2t, sim_i2t_targets, imt_mask):
    """
    稳健计算 loss，避免 -inf * 0 导致的 NaN。
    sim_i2t: Tensor shape (batch_k, n) -- logits 相似度
    sim_i2t_targets: Tensor shape (batch_k, n) -- multi-hot / 回归目标（与 sim_i2t 对齐）
    imt_mask: Tensor shape (orig_batch_k, n_orig) -- 全局 mask，需切片到 sim_i2t 尺寸

    返回 scalar loss
    """
    # 确保类型一致为 float
    sim_i2t = sim_i2t.float()
    sim_i2t_targets = sim_i2t_targets.float()
    # 截取 mask 到 sim_i2t 的尺寸
    mask = imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]].float()

    # 检查并替换掉输入中的 NaN/Inf（可选，但推荐）
    if torch.isnan(sim_i2t).any() or torch.isinf(sim_i2t).any():
        # 将极端值裁切为有限数（根据需要调整 clip 值）
        sim_i2t = torch.nan_to_num(sim_i2t, nan=0.0, posinf=1e9, neginf=-1e9)

    # 计算 log_probs
    log_probs = F.log_softmax(sim_i2t, dim=1)  # 形状 (batch_k, n)

    # 构造有效位置 mask：只有当目标和掩码都为 1 时才视为有效
    effective_mask = (sim_i2t_targets > 0) & (mask > 0)
    effective_mask = effective_mask.to(log_probs.dtype)  # float 0/1

    # 把无效位置的 log_probs 置为 0，这样后续乘以 0 不会产生 -inf * 0 的情况
    # （因为我们只在有效位置累加 log_probs）
    safe_log_probs = torch.where(effective_mask.bool(), log_probs, torch.zeros_like(log_probs))

    # 每行累加有效位置的 log_prob（注意乘上 effective_mask 以保留权重为非 1 的情况）
    per_row_sum = (safe_log_probs * effective_mask).sum(dim=1)  # 形状 (batch_k,)

    # 每行有效数量（用于按行归一化，避免不同样本权重不一致）
    counts = effective_mask.sum(dim=1).clamp_min(1.0)  # 防止除以 0

    # 按行平均再求批平均
    loss = -(per_row_sum / counts).mean()
    return loss

def compute_mlm3(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)

    # loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])\
    
    
    # *imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]]
    sim_i2t = infer["local_image"] @ infer["local_text_m_all"] / pl_module.temp
    imt_mask = pl_module.imt_mask.to(sim_i2t.device)
    if torch.isnan(sim_i2t).any() or torch.isinf(sim_i2t).any():
        # 将极端值裁切为有限数（根据需要调整 clip 值）
        sim_i2t = torch.nan_to_num(sim_i2t, nan=0.0, posinf=1e9, neginf=-1e9)
    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*infer["sim_i2t_targets"]*imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]],dim=1).mean()

    # loss_i2t = stable_sim_loss(sim_i2t, infer["sim_i2t_targets"], imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]])
    # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*infer["sim_t2i_targets"]*imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]],dim=1).mean() 

    #print((F.log_softmax(sim_i2t, dim=1)*infer["sim_i2t_targets"]).shape, loss_t2i.shape) 30, 1030

    loss_ita = loss_i2t
    # loss_local, loss_local2, attn = local_loss(infer['local_image_word'].squeeze(0).permute(0,2,1), infer['local_text_word'].squeeze(0).permute(0,2,1), [10 for l in range(1000)])

    if torch.isnan(loss_ita):
        loss_ita = infer["local_image"].sum() * 0
        print("loss is nan")

    ret = {
        "mlm_loss": loss_ita, # + loss_ita, # (loss_local + loss_local2) / 2 * 0.1 + , # +  , # + loss_cls * 0.3,
        #"mlm_logits": mlm_logits,
        #"mlm_labels": mlm_labels,
        # "mlm_ids": infer["text_ids"],
    }

    ### cal the retrieval accuracy
    local_text = infer["local_text"].reshape(-1, 10, pl_module.aligned_length)
    local_image = infer["local_image"].reshape(-1, 10, pl_module.aligned_length)
    

    # print(local_text.shape, local_image.shape, pl_module.test_text_features.shape)
    acc = 0
    if not pl_module.training:
        for l in range(local_text.shape[0]):
            test_text_feature = torch.cat([local_text[l:l+1].cpu(), pl_module.test_text_features], 0)
            sim = torch.topk(F.cosine_similarity(local_image[l:l+1].cpu(), test_text_feature, dim=-1).sum(-1), 50).indices
            # print(sim)
            if 0 in sim:
                acc += 1
    
    
    acc = acc / local_text.shape[0]

    # print(sim.shape, acc)
    # print(sim)


    phase = "train" if pl_module.training else "val"
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(acc)
    pl_module.log(f"mlm/{phase}/loss", loss_ita)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    
    return ret

def compute_mlm4(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)

    loss_cls = torch.nn.BCEWithLogitsLoss()(infer["logits"], infer["obver_label"].float())

    pred_abnormality = (torch.sigmoid(infer["logits"]) > 0.5).cpu().numpy()
    
    obver_label = infer["obver_label"].cpu().numpy()
    # print(infer["logits"])
    # print(obver_label.shape)

    cf1 = []
    for ii in range(obver_label.shape[0]):
        # print(classification_report(obver_label[ii].flatten(), pred_abnormality[ii].flatten(), output_dict=True))
        cf1.append(classification_report(obver_label[ii].flatten(), pred_abnormality[ii].flatten(), output_dict=True).get('1', {}).get('f1-score', 0.0))
    cf1 = np.mean(cf1)

    # print("current F1: ", cf1)
    # *imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]]
    sim_i2t = infer["local_image"] @ infer["local_text_m_all"] / pl_module.temp
    imt_mask = pl_module.imt_mask.to(sim_i2t.device)
    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*infer["sim_i2t_targets"]*imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]],dim=1).mean()
    # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*infer["sim_t2i_targets"]*imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]],dim=1).mean() 

    loss_ita = loss_i2t + loss_cls


    # loss_local, loss_local2, attn = local_loss(infer['local_image_word'].squeeze(0).permute(0,2,1), infer['local_text_word'].squeeze(0).permute(0,2,1), [10 for l in range(1000)])

    if torch.isnan(loss_ita):
        loss_ita = infer["local_image"].sum() * 0
        print("fxxk")

    ret = {
        "mlm_loss": loss_ita, # + loss_ita, # (loss_local + loss_local2) / 2 * 0.1 + , # +  , # + loss_cls * 0.3,
        #"mlm_logits": mlm_logits,
        #"mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    ### cal the retrieval accuracy
    local_text = infer["local_text"].reshape(-1, 10, pl_module.aligned_length)
    local_image = infer["local_image"].reshape(-1, 10, pl_module.aligned_length)
    

    # print(local_text.shape, local_image.shape, pl_module.test_text_features.shape)
    acc = 0
    if not pl_module.training:
        for l in range(local_text.shape[0]):
            test_text_feature = torch.cat([local_text[l:l+1].cpu(), pl_module.test_text_features], 0)
            sim = torch.topk(F.cosine_similarity(local_image[l:l+1].cpu(), test_text_feature, dim=-1).sum(-1), 50).indices
            # print(sim)
            if 0 in sim:
                acc += 1
    
    accr = round((acc / local_text.shape[0] * 10000))
    
    # print(acc)

    acc = accr + cf1

    # if not pl_module.training:
    #     print(accr, cf1)

    # print(sim.shape, acc)
    # print(sim)


    phase = "train" if pl_module.training else "val"
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(acc)
    pl_module.log(f"mlm/{phase}/loss", loss_ita)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    
    return ret


def compute_mlm5(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)

    # loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])\
    
    if False:
        # *imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]]
        sim_i2t = infer["local_image"] @ infer["local_text_m_all"] / pl_module.temp
        
        sim_i2t = pl_module._per_organ(sim_i2t, infer["bs"])
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=2)*infer["sim_i2t_targets"],dim=-1).mean()
        # print(sim_i2t.shape)
        if torch.isnan(sim_i2t).any() or torch.isinf(sim_i2t).any():
            # 将极端值裁切为有限数（根据需要调整 clip 值）
            sim_i2t = torch.nan_to_num(sim_i2t, nan=0.0, posinf=1e9, neginf=-1e9)
        # print(sim_i2t.shape)
        # print()
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=2)*infer["sim_i2t_targets"],dim=-1).mean()
    else:
        # 修改后的稳定计算流程
        local_image = infer["local_image"]
        bad = torch.isnan(local_image) | torch.isinf(local_image)
        if bad.sum() > 0:
            # 填充 0 并 detach，梯度畅通
            local_image = torch.where(bad, torch.zeros_like(local_image).detach(), local_image)
            print("localiamge nan ", bad.sum())
            exit(0)

        raw_sim = local_image @ infer["local_text_m_all"]

        # local_image = infer["local_image"]
        bad = torch.isnan(raw_sim) | torch.isinf(raw_sim)
        if bad.sum() > 0:
            # 填充 0 并 detach，梯度畅通
            raw_sim = torch.where(bad, torch.zeros_like(raw_sim).detach(), raw_sim)
            print("simnan")

    
        raw_sim =  pl_module._per_organ(raw_sim, infer["bs"]) # 进行一些reshape操作
        # 数值稳定处理（保持梯度）
        max_val = raw_sim.detach().max(dim=2, keepdim=True).values.detach()
        stable_sim = raw_sim - max_val  # 平移数值范围

        # 温度缩放
        sim_i2t = stable_sim / pl_module.temp # 0.07

        # 自定义操作
        # sim_i2t = pl_module._per_organ(sim_i2t, infer["bs"])

        # 防御性检查
        if torch.isnan(sim_i2t).any():
            print(raw_sim.max(), raw_sim.min())
            print(F.log_softmax(sim_i2t, dim=2).max(), F.log_softmax(sim_i2t, dim=2).min())
            print(f"[Error] NaN出现！输入范围: [{stable_sim.min().item():.4f}, {stable_sim.max().item():.4f}]")
            # 保留计算图的同时处理异常值
            sim_i2t = torch.where(torch.isnan(sim_i2t), torch.zeros_like(sim_i2t), sim_i2t)
            exit(0)

        # 损失计算
        log_probs = F.log_softmax(sim_i2t, dim=2)
        loss_i2t = -torch.sum(log_probs * infer["sim_i2t_targets"], dim=-1).mean()
    # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*infer["sim_t2i_targets"]*imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]],dim=1).mean() 

    #print((F.log_softmax(sim_i2t, dim=1)*infer["sim_i2t_targets"]).shape, loss_t2i.shape) 30, 1030
    # print(infer['mask_loss'])
    loss_ita = loss_i2t + infer['mask_loss'] #  * 0.1
    # print(loss_ita)

    # loss_local, loss_local2, attn = local_loss(infer['local_image_word'].squeeze(0).permute(0,2,1), infer['local_text_word'].squeeze(0).permute(0,2,1), [10 for l in range(1000)])

    if torch.isnan(loss_ita):
        loss_ita = infer["local_image"].sum() * 0
        print("loss is nan")

    ret = {
        "mlm_loss": loss_ita, # + loss_ita, # (loss_local + loss_local2) / 2 * 0.1 + , # +  , # + loss_cls * 0.3,
        #"mlm_logits": mlm_logits,
        #"mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    ### cal the retrieval accuracy
    local_text = infer["local_text"].reshape(-1, 10, pl_module.aligned_length)
    local_image = infer["local_image"].reshape(-1, 10, pl_module.aligned_length)
    

    # print(local_text.shape, local_image.shape, pl_module.test_text_features.shape)
    acc = 0
    if not pl_module.training:
        for l in range(local_text.shape[0]):
            test_text_feature = torch.cat([local_text[l:l+1].cpu(), pl_module.test_text_features], 0)
            sim = torch.topk(F.cosine_similarity(local_image[l:l+1].cpu(), test_text_feature, dim=-1).sum(-1), 50).indices
            # print(sim)
            if 0 in sim:
                acc += 1
    
    
    acc = acc / local_text.shape[0]

    # print(sim.shape, acc)
    # print(sim)


    phase = "train" if pl_module.training else "val"
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(acc)
    pl_module.log(f"mlm/{phase}/loss", loss_ita)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    
    return ret

def compute_mlm5_onlycls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)

    loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])


    probabilities = torch.sigmoid(infer["norm_class_res"])

    # 将概率转换为二进制预测（0或1）
    predictions = (probabilities >= 0.5).float()

    # 计算准确率
    correct_predictions = (predictions == infer["norm_class_label"]).float()
    accuracy = correct_predictions.sum() / torch.ones_like(infer["norm_class_label"]).sum()

    # print(accuracy)

    ret = {
        "mlm_loss": loss_cls, # (loss_local + loss_local2) / 2 * 0.1 + , # +  , # + loss_cls * 0.3,
        #"mlm_logits": mlm_logits,
        #"mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    acc = accuracy # getattr(pl_module, f"{phase}_mlm_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    pl_module.log(f"mlm/{phase}/loss", loss_cls)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    
    return ret

def compute_mim(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)

    if pl_module.hparams.config["mim_layer"] == -1:
        multi_modal_image_feats = infer["multi_modal_image_feats"]
    else:
        layer_idx = pl_module.hparams.config["mim_layer"]
        multi_modal_image_feats = infer[f"multi_modal_image_feats_{layer_idx}"]

    mim_logits = pl_module.mim_head(multi_modal_image_feats, infer["mim_ids_restore"])

    target = infer["patched_images"]
    if pl_module.hparams.config["norm_pix_loss"]:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
    mim_labels = target
    mask = infer["mim_masks"]

    mim_loss = (mim_logits - mim_labels) ** 2
    mim_loss = mim_loss.mean(dim=-1)  # [N, L], mean loss per patch
    mim_loss = (mim_loss * mask).sum() / mask.sum()  # mean loss on removed patches

    ret = {
        "mim_loss": mim_loss,
        "mim_logits": mim_logits,
        "mim_labels": mim_labels,
    }
    

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mim_loss")(ret["mim_loss"])
    acc = -loss
    pl_module.log(f"mim/{phase}/loss", loss)
    pl_module.log(f"mim/{phase}/accuracy", acc)

    return ret


def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_head(infer["multi_modal_cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_vqa(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_head(infer["multi_modal_cls_feats"])
    vqa_targets = torch.zeros(len(vqa_logits), pl_module.hparams.config["vqa_label_size"]).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]
    vqa_answer_types = torch.tensor(batch["answer_types"]).to(pl_module.device)

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1])

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
        "vqa_answer_types": vqa_answer_types,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(ret["vqa_logits"], ret["vqa_targets"], ret["vqa_answer_types"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_cls(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    cls_logits = pl_module.cls_head(infer["multi_modal_cls_feats"])
    cls_labels = batch["cls_labels"]
    cls_loss = F.cross_entropy(cls_logits, cls_labels)

    ret = {
        "cls_loss": cls_loss,
        "cls_logits": cls_logits,
        "cls_labels": cls_labels,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_cls_loss")(ret["cls_loss"])
    acc = getattr(pl_module, f"{phase}_cls_accuracy")(ret["cls_logits"], ret["cls_labels"])
    pl_module.log(f"cls/{phase}/loss", loss)
    pl_module.log(f"cls/{phase}/accuracy", acc)

    return ret


def compute_irtr(pl_module, batch, test=False):
    is_training_phase = pl_module.training
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack([batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = torch.stack([batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = torch.stack([batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    batch_infer = {
        "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
        "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
        "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
        "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
    }

    infer = pl_module.infer(batch_infer)

    score = pl_module.irtr_head(infer["multi_modal_cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {"irtr_loss": irtr_loss}

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=256,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(text_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(image_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    # TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # == Begin: Add New Keys ==
        batch_text_preload = {
            "text_ids": _b["text_ids"].to(pl_module.device),
            "text_masks": _b["text_masks"].to(pl_module.device),
            "text_labels": _b["text_labels"].to(pl_module.device),
            "img_index": _b["img_index"],
        }
        text_preload.append(batch_text_preload)
        # == End  : Add New Keys ==

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                # == Begin: Add New Keys ==
                batch_infer = {
                    "text_ids": txt_batch["text_ids"],
                    "text_masks": txt_batch["text_masks"],
                    "text_labels": txt_batch["text_labels"],
                }
                score = pl_module.irtr_head(pl_module.infer(batch_infer, img=im, )["multi_modal_cls_feats"])[:, 0]
                # == End  : Add New Keys ==

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)
