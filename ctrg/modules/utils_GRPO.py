from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys, time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

# for metric
from .online_utils import get_results
from sklearn.metrics import classification_report
from ..gadgets.my_metrics import Accuracy, Scalar, VQARADScore

from bottle import request
import bottle, threading, queue
from .online_utils import *
import evaluate
from nltk import download
from nltk.tokenize import sent_tokenize
from paths_config import BLEU_METRIC_PATH, GRPO_REWARD_SERVER_URL
download('punkt')      # BLEU / METEOR 需要


def infer_text(text_model, input_ids, maps, device, aligned_length=512):
    """Encode structure-wise text with a frozen text model; return CLS tokens.

    Relocated here (from GRPO/ref_LlaMa32_server_gen.py) so that both the GRPO
    client model and the servers can import `seg_text` without triggering the
    server module's heavy top-level imports.
    """
    input_ids = input_ids.reshape(-1, input_ids.shape[-1]).to(device)
    maps = maps.reshape(-1, input_ids.shape[-1]).to(device)
    with torch.no_grad():
        local_text_ori = text_model(input_ids, attention_mask=maps)[0][:, 0:1, :]
    return local_text_ori


def seg_text(text, model, tokenizer):
    """Split a report into the 10 anatomical-structure segments and encode each.

    Keyword-matching rule identical to the dataset / server implementation.
    Returns the per-structure CLS embeddings via `infer_text`.
    """
    device = next(model.parameters()).device
    follow_segtence = []

    keywords = [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"],
                ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"],
                ["Pleura", "thorax", "membrane", "diaphragm"],
                [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"],
                ["thyroid"],
                ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"],
                ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]

    segs = sent_tokenize(text)
    idx_label = [[] for _ in segs]

    for seg_idx, seg in enumerate(segs):
        for keyidx, keyword in enumerate(keywords):
            for key in keyword:
                if key.lower() in seg.lower():
                    idx_label[seg_idx].append(keyidx)
                    break

    you = False
    for iidx in range(len(idx_label)):
        if idx_label[iidx] == []:
            s = segs[iidx].strip().lower()
            if s[:3] == "and" or s[:11] == "the largest" or s[:11] == "in addition" or s[:14] == "the appearance" or s[:7] == "however" or s[:5] == "again" or s[:2] == "it" or s[:4] == "with" or s[:11] == "surrounding" or s[:11] == "the nodules" or s[:5] == "which" or s[:5] == "about" or s[:10] == "especially" or s[:5] == "after" or s[:5] == "signs" or s[:4] == "left" or s[:8] == "the size" or s[:4] == "size" or s[:8] == "ct value" or s[:11] == "ct diameter":
                idx_label[iidx] = idx_label[iidx - 1]
            if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                idx_label[iidx] = idx_label[iidx - 1]
            elif iidx > 0 and segs[iidx - 1].strip()[-1] == ",":
                idx_label[iidx] = idx_label[iidx - 1]
            elif segs[iidx] in follow_segtence:
                idx_label[iidx] = idx_label[iidx - 1]
            elif (iidx != 0 and iidx != len(idx_label) - 1):
                for kkk in range(iidx + 1, len(idx_label)):
                    if idx_label[iidx - 1] == idx_label[kkk]:
                        idx_label[iidx] = idx_label[iidx - 1]
                        break
                    elif idx_label[kkk] != []:
                        break
            else:
                if not you:
                    continue
        else:
            you = True

    for iidx in range(1, len(idx_label) - 1):
        if 1 in idx_label[iidx] and (1 in idx_label[iidx - 1] or 1 in idx_label[iidx + 1]):
            idx_label[iidx] = [1]
        elif 2 in idx_label[iidx] and (2 in idx_label[iidx - 1] or 2 in idx_label[iidx + 1]):
            idx_label[iidx] = [2]
        elif 3 in idx_label[iidx] and (3 in idx_label[iidx - 1] or 3 in idx_label[iidx + 1]):
            idx_label[iidx] = [3]
        elif 0 in idx_label[iidx]:
            idx_label[iidx] = [0]

    final_seg = ["" for _ in range(10)]
    for iidx in range(len(idx_label)):
        if idx_label[iidx] is []:
            final_seg[9] += (" " + segs[iidx])
            continue
        for count, tidx in enumerate(list(set(idx_label[iidx]))):
            if tidx < 9:
                final_seg[tidx] += (" " + segs[iidx] + ",")
            elif len(list(set(idx_label[iidx]))) == 1:
                final_seg[9] += (" " + segs[iidx])

    defaults = ["No abnormality found in trachea.",
                "No abnormality found in mediastinum and heart.",
                "No abnormality found in lung.",
                "No abnormality found in esophagus.",
                "No abnormality found in pleural.",
                "No abnormalities in rib.",
                "No abnormalities in thyroid.",
                "No abnormalities in chest.",
                "No abnormalities in abdomen organs.",
                "No abnormalities in other organs."]
    for iidx, seg in enumerate(final_seg):
        if len(seg) == 0:
            final_seg[iidx] += defaults[iidx] if iidx == 0 else (" " + defaults[iidx])

    all_encodings = []
    for seg in final_seg:
        all_encodings.append(tokenizer(seg, padding="max_length", truncation=True,
                                       max_length=200, return_special_tokens_mask=True,
                                       return_tensors="pt"))
    all_encodings_i = torch.cat([l.input_ids for l in all_encodings])
    all_maps_i = torch.cat([l.attention_mask for l in all_encodings])
    return infer_text(model, all_encodings_i, all_maps_i, device)

def reward_think(think, img_fea_path, prior, label, use_green = False):
    data = {
        'base': json.dumps(str(use_green)),
        'think': json.dumps(think),
        'prior': json.dumps(prior),  # Encode the string to bytes
        "img":json.dumps(img_fea_path),
    }

    data_list = [
        data['base'].encode('utf-8'),
        data["img"].encode('utf-8'),
        data['think'].encode('utf-8'),
        data['prior'].encode('utf-8'),
    ]

    upload_data = make_bytes_list(data_list)

    #print(upload_data)

    # Send data to the server
    upload_url = f'{GRPO_REWARD_SERVER_URL}/upload'
    response = requests.post(upload_url, data=upload_data)
    request_id = response.content.decode('utf-8')
    # print("Upload response:", response.content)

    # Retrieve result from the server
    get_url = f'{GRPO_REWARD_SERVER_URL}/get/{request_id}'
    response = requests.get(get_url)
    while response.content == b'empty':
        response = requests.get(get_url)
        # print("waiting", end = "\r")
    result_data = bytes_list_to_list(response.content)
    result_tensor = bytes_to_tensor(result_data[1])
    result_report = json.loads(result_data[2].decode('utf-8'))

    # return result_report
    if not use_green:
        f1s = []
        for idx, result in enumerate(result_tensor):
            f1s.append(classification_report(label[idx % len(label)].flatten(), result.flatten(), output_dict=True).get('1', {}).get('f1-score', 0.0))
    else:
        f1s = result_tensor

    return f1s, result_report


def reward_blue(answer, label, item):
    
    # print(item, answer)
    bleu = evaluate.load(BLEU_METRIC_PATH)
    # rouge = evaluate.load('/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/metric_cal/rough/rouge.py')
    # meteor = evaluate.load('/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/metric_cal/metero/meteor.py')

    BLEU4 = bleu.compute(predictions=[item], references=[answer], max_order=4)['bleu']
    # print(BLEU[4])
    return BLEU4

def reward_format(answer):
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    # pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    # pattern = r"^<think>.*?</think><answer>.*?</answer>"
    pattern = r"^<think>.*?</think>(?: |\. )?<answer>.*?</answer>.*"

    return 1.25 if re.match(pattern, answer, re.DOTALL) else -1

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    # print(logits.shape)
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def GRPO_step(batch, device, tokenizer, logits):
    beta = 0.04
    clip_param = 0.2

    prompt_length = batch['plen']
    inputs = batch['inputs'].to(device)
    advantages = batch['rewards'].to(device).unsqueeze(1)   # normalized in generation

    # logits = engine(inputs_embeds = inputs).logits
    logits = logits[:, prompt_length:-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = batch['completion_ids'][:, 1:].clone()  # (B, L-1), exclude the first input ID since we don't have logits for it
    
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps # [:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    # print(ref_per_token_logps.shape, per_token_logps.shape)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (batch['completion_ids'][:, 1:] != tokenizer.pad_token_id).int()

    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(device))

        # print(ratio)
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages

    # per_token_loss = -(per_token_loss - beta * per_token_kl)
    # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # loss_per_sample = ((per_token_loss - beta * per_token_kl) * completion_mask).sum(dim=1)
    # loss = loss_per_sample.mean()   # 每條樣本等權
    loss_per_sample = -((per_token_loss - beta * per_token_kl) * completion_mask).sum(dim=1)
    loss = loss_per_sample.mean()

    # per_token_loss = -(per_token_loss - beta * per_token_kl)
    # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    # loss.backward(retain_graph=True)
    # g = per_token_logps.grad  # 應該不為 None 且 norm > 0
    # print(g.norm().item())
    return loss