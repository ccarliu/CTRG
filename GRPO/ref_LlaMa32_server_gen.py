import json, os, shutil, re, random, io, time
import torch
import numpy as np

from peft import get_peft_model, LoraConfig, TaskType
import uuid
import nltk
import copy
from transformers import RobertaConfig, RobertaModel, BertTokenizer, AutoConfig
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctrg.config import ex
# nltk.download('punkt')  # 下载punkt模型，用于句子分割

from nltk.tokenize import sent_tokenize

from ctrg.modules.report_generation_v38_llm_v3 import M3AETransformerSS_3D_lmae_rg_v38_llm_v3

from ctrg.modules.online_utils import get_results, get_results_green
from paths_config import (
    BIOMEDVLP_BERT, GRPO_REWARD_SERVER_PORT, require,
    GRPO_DFULL_CKPT, GRPO_DFULL_GROUP_SIZE, GRPO_DFULL_K_CUR,
    CTRATE_TRAIN_REPORTS, CTRATE_VALID_REPORTS,
    FEATURE_PATH_TRAIN, FEATURE_PATH_VALID,
)
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.cuda.amp import autocast
from transformers import LlamaTokenizer

import csv

# use_green = True

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

def infer_text(text_model, input_ids, maps, device, aligned_length = 512):
        
    # print(text_seg_index)
    #print(index.shape)
    input_ids = input_ids.reshape(-1, input_ids.shape[-1]).to(device)
    maps = maps.reshape(-1, input_ids.shape[-1]).to(device)
    # print(text_model.device, device)
    with torch.no_grad():
        
        local_text_ori = text_model(input_ids, attention_mask = maps)[0][:, 0:1, :] # .permute(1,0,2)
            # print(seg)

    
    return local_text_ori #local_text_ori

def seg_text(text, model, tokenizer):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        ###### pre seg
        segs = sent_tokenize(text)
        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == []:

                #print(xxx)
                # print(segs[iidx], segs[iidx-1].strip()[-1], iidx)
                #print(segs[iidx].strip()[:11])
                if segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]


                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]

                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += " " + ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += " " + ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += " " + ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += " " + ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += " " + ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += " " + ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += " " + ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += " " + ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += " " + ("No abnormalities in other organs.")
            
                continue

        # print(final_seg)
        all_encodings = []
        for seg in final_seg:
            all_encodings.append(tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))

        all_encodings_i = torch.cat([l.input_ids for l in all_encodings])
        all_maps_i = torch.cat([l.attention_mask for l in all_encodings])
        # print(all_encodings_i, all_encodings_i.float().mean(1))

        # infer_text(model, all_encodings_i, all_maps_i, device)# text_model, input_ids, maps, device
        return infer_text(model, all_encodings_i, all_maps_i, device)# text_model, input_ids, maps, device

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    require("BIOMEDVLP_BERT", "LLM_DECODER_PATH", "GRPO_DFULL_CKPT",
            "FEATURE_PATH_TRAIN", "FEATURE_PATH_VALID")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    #####################
    textmodelpath = BIOMEDVLP_BERT
    tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

    # path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-2-7b-chat-hf"
    #path = "/apdcephfs_cq10/share_1290796/lh/dataset/Meta-Llama-3-8B-Instruct"

    # tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
    # tokenizer.pad_token_id = 0
    #self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    #####################

    _config["image_size"] = [480, 480]
    _config["image_depth"] = 240
    _config["batch_size"] = 1
    _config["per_gpu_batchsize"] = 1
    _config["selected_patch"] = 9
    ref_model = M3AETransformerSS_3D_lmae_rg_v38_llm_v3(_config)

    print('Loading LLAMA LoRA Done') 

    # Report generator D_full (Stage-3 train_mode=full checkpoint).
    ckpath = GRPO_DFULL_CKPT
    if not (ckpath and os.path.exists(ckpath)):
        raise FileNotFoundError(f"GRPO_DFULL_CKPT not found: '{ckpath}' (Stage-3 D_full checkpoint)")
    # AP-LTF selection used when D_full generates the report: M candidate
    # patches per structure, keep k_cur of them (values used in the paper runs).
    Group_s = GRPO_DFULL_GROUP_SIZE
    k_cur = GRPO_DFULL_K_CUR

    ck = torch.load(ckpath)["state_dict"]
    
    ref_model.load_state_dict(ck, strict = False)
    print("load pretrained weights from: ", ckpath)

    ref_model = ref_model.to(device)
    ref_model.eval()

    text_path = CTRATE_VALID_REPORTS
    files = open(text_path, "r")
    reader = csv.reader(files)
    all_text = {}
    for cline in reader:
        if cline[3] == "Findings_EN":
            continue
        
        if cline[0].split("_")[-1][:1] != "1":
            continue
        
        all_text[cline[0].split(".")[0]] = cline[3]

    text_path = CTRATE_TRAIN_REPORTS
    files = open(text_path, "r")
    reader = csv.reader(files)
    all_text_train = {}
    for cline in reader:
        if cline[3] == "Findings_EN":
            continue
        
        if cline[0].split("_")[-1][:1] != "1":
            continue
        
        all_text_train[cline[0].split(".")[0]] = cline[3]

    # print(next(ref_model.text_decoder.parameters()).dtype)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def load_img(img_path):
        feature_path_t = FEATURE_PATH_TRAIN
        feature_path_v = FEATURE_PATH_VALID

        # print(img_path)
        if "valid" in img_path:
            all_feature = torch.tensor(np.load(os.path.join(feature_path_v,  img_path.replace("selected_patch", "image_feature").replace("image_feature.npy", "") + "image_feature.npy")))
        else:
            all_feature = torch.tensor(np.load(os.path.join(feature_path_t,  img_path.replace("selected_patch", "image_feature").replace("image_feature.npy", "") + "image_feature.npy")))
        
        # print("load img: ", all_feature.shape)
        return all_feature
    
    def load_gts(img_path):
        if "valid" in img_path:
            return all_text[img_path.split("/")[-1].split(".")[0]]
        else:
            return all_text_train[img_path.split("/")[-1].split(".")[0]]

    def get_per_token_logps(select_patch_path, thinks, prior, use_green = False):
        D = []
        for think in thinks:
            D.append(seg_text(think, ref_model.text_model, tokenizer).squeeze().unsqueeze(0))
        D = torch.cat(D, 0)
        
        uni_modal_image_feats = [load_img(cpath) for cpath in select_patch_path]
        gts = [load_gts(cpath) for cpath in select_patch_path]

        if len(uni_modal_image_feats) == 1: 
            uni_modal_image_feats = uni_modal_image_feats[0].cuda().repeat(D.shape[0], 1, 1)
            gts = gts * D.shape[0]
        else:
            uni_modal_image_feats = torch.cat(uni_modal_image_feats, 0).cuda()
        
        # print(uni_modal_image_feats.shape, D.shape)
        # with torch.no_grad():
        selected_patch = ref_model.infer_image_(ref_model.expert_image, D, uni_modal_image_feats, ref_model.vision_extract_layer, M = Group_s, k_cur=k_cur)

        selected_patch = ref_model.multi_modal_vision_proj_forllm(selected_patch)

        image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                 device=device)
        
        # print(selected_patch.shape, image_masks.shape, len(prior * D.shape[0]))
        if len(prior) != D.shape[0]:
            img_embeds, atts_img = ref_model.prompt_wrap(selected_patch, image_masks, prior * D.shape[0])
        else:
            img_embeds, atts_img = ref_model.prompt_wrap(selected_patch, image_masks, prior)

        '''
        with autocast():
            outputs = ref_model.text_decoder.generate(
                    inputs_embeds=img_embeds.float(),
                    num_beams=1,
                    do_sample=False,
                    min_new_tokens=50,
                    max_new_tokens=420,
                    return_dict_in_generate=True, 
                    # output_attentions=True,
                )
        '''
        outputs = ref_model.text_decoder.generate(
            inputs_embeds=img_embeds.half(),
            attention_mask = atts_img.half(),
            num_beams=1,
            do_sample=False,
            min_new_tokens=50,
            max_new_tokens=450,
            return_dict_in_generate=True, 
            repetition_penalty=1.05,
            pad_token_id= ref_model.tokenizer.eos_token_id # 128001  # 或者使用其他合适的pad_token_id
            # output_attentions=True,
        )

        captions = []
        for output in outputs.sequences:
            caption = ref_model.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[:])

        ###########################
        # print(captions)
        if not use_green:
            pred_abnormality = get_results(captions)
        else:
       
            pred_abnormality = get_results_green(captions, gts)
        # print(pred_abnormality)
        return torch.tensor(pred_abnormality), captions

    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()
    result_dict = {}

    app = bottle.Bottle()
    logging.getLogger('bottle').setLevel(logging.CRITICAL)
    logging.getLogger('tornado.access').setLevel(logging.WARNING)



    @app.route('/upload', method='POST')
    def do_upload():
        request_id = str(uuid.uuid4())  # 生成唯一请求ID
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        # print(dd)
        data = {'base': json.loads(dd[0])} 
        data['img'] = json.loads(dd[1].decode('utf-8'))
        data['think'] = json.loads(dd[2].decode('utf-8'))
        data['prior'] = json.loads(dd[3].decode('utf-8'))
        data['id'] = request_id

        raw_queue.put(data)
        # print(data)
        # print('receive: ', data['img'], data['think']) # , data['img'])
        return request_id.encode('utf-8')

    @app.route('/get/<request_id>', method='GET')
    def do_get(request_id):
        if request_id not in result_dict:
            return b'empty'
        result = result_dict.pop(request_id)  # 获取并移除结果
        return result
    
    def run_server(): bottle.run(app, host='0.0.0.0', port=GRPO_REWARD_SERVER_PORT, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()
    ii = 0
    while True:
        d = raw_queue.get()
        request_id = d['id']
        with torch.inference_mode():
            abnorm_pred_results, captions = get_per_token_logps(d['img'], d['think'], d['prior'], eval(d['base']))
        ii += 1
        # if ii % 605 == 0:
        #     if os.path.exists("/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/results/task_pretrain_m3ae-seed0-from_/version_256/checkpoints/last.ckpt"):
        #         ck = torch.load(ckpath)["state_dict"]
    
        #         ref_model.load_state_dict(ck)
        ###############################
        # per_token_logps = per_token_logps[:,prompt_length-1:]
        data = [json.dumps(d['base']).encode(), 
                tensor_to_bytes(abnorm_pred_results), json.dumps(captions).encode()]
        xdata = make_bytes_list(data)
        result_dict[request_id] = xdata

    