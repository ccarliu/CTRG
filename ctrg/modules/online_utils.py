import requests
import json
import torch
import io
import time
from nltk.tokenize import sent_tokenize
from paths_config import GRPO_F1_SERVER_URL, GRPO_GREEN_SERVER_URL
from transformers import LogitsProcessor, LogitsProcessorList

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

def get_results(sentence_input):
    # Prepare data for upload
    data = {
        'base': json.dumps({'example_key': 'example_value'}),
        'inputs': json.dumps(sentence_input),  # Encode the string to bytes
    }

    data_list = [
        data['base'].encode('utf-8'),
        data['inputs'].encode('utf-8'),
    ]

    upload_data = make_bytes_list(data_list)

    #print(upload_data)

    # Send data to the server
    upload_url = f'{GRPO_F1_SERVER_URL}/upload'
    response = requests.post(upload_url, data=upload_data)
    request_id = response.content.decode('utf-8')
    # print("Upload response:", response.content)

    # Retrieve result from the server
    get_url = f'{GRPO_F1_SERVER_URL}/get/{request_id}'
    response = requests.get(get_url)
    while response.content == b'empty':
        response = requests.get(get_url)
        # print("waiting", end = "\r")
    result_data = bytes_list_to_list(response.content)
    result_tensor = bytes_to_tensor(result_data[0])
    
    return result_tensor

def get_results_green(sentence_input, gts):
    # Prepare data for upload
    data = {
        'base': json.dumps({'example_key': 'example_value'}),
        'inputs': json.dumps(sentence_input),  # Encode the string to bytes
        'gts': json.dumps(gts),  # Encode the string to bytes
    }

    data_list = [
        data['base'].encode('utf-8'),
        data['inputs'].encode('utf-8'),
        data['gts'].encode('utf-8'),
    ]

    upload_data = make_bytes_list(data_list)

    #print(upload_data)

    # Send data to the server
    upload_url = f'{GRPO_GREEN_SERVER_URL}/upload'
    response = requests.post(upload_url, data=upload_data)
    request_id = response.content.decode('utf-8')
    # print("Upload response:", response.content)

    # Retrieve result from the server
    get_url = f'{GRPO_GREEN_SERVER_URL}/get/{request_id}'
    response = requests.get(get_url)
    while response.content == b'empty':
        response = requests.get(get_url)
        # print("waiting", end = "\r")
    result_data = bytes_list_to_list(response.content)
    result_tensor = bytes_to_tensor(result_data[0])
    
    return result_tensor


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
                max_length=150,
                return_special_tokens_mask=True,
                return_tensors="pt"))

        all_encodings_i = torch.cat([l.input_ids for l in all_encodings])
        all_maps_i = torch.cat([l.attention_mask for l in all_encodings])


        # infer_text(model, all_encodings_i, all_maps_i, device)# text_model, input_ids, maps, device
        return infer_text(model, all_encodings_i, all_maps_i, model.device)# text_model, input_ids, maps, device


class BeamableRealtimeContrastProcessor(LogitsProcessor):
    def __init__(self, decoder,
                 img_embeds, atts_img,
                 img_embeds_contra, atts_img_contra,
                 beta=0.5, num_beams=1):
        super().__init__()
        self.decoder = decoder
        self.beta = beta
        self.num_beams = num_beams
        B, L_img, _ = img_embeds.shape
        device = img_embeds.device

        # 1. 前缀 forward，拿到 KV-Cache 和最后一个位置的 logits
        with torch.no_grad():
            out_main = decoder(
                inputs_embeds=img_embeds,
                attention_mask=atts_img,
                use_cache=True,
                return_dict=True
            )
            # 复制到 beam 维
            self.past_main = self._expand_kv(out_main.past_key_values, num_beams)
            self.init_logits_main = out_main.logits[:, -1, :] \
                .repeat_interleave(num_beams, dim=0)          # [B*beam, V]

            out_contra = decoder(
                inputs_embeds=img_embeds_contra,
                attention_mask=atts_img_contra,
                use_cache=True,
                return_dict=True
            )
            self.past_contra = self._expand_kv(out_contra.past_key_values, num_beams)
            self.init_logits_contra = out_contra.logits[:, -1, :] \
                .repeat_interleave(num_beams, dim=0)          # [B*beam, V]

        # 2. mask 也一次性复制到 B*beam
        self.mask_main   = atts_img.repeat_interleave(num_beams, dim=0)    # [B*beam, L_img]
        self.mask_contra = atts_img_contra.repeat_interleave(num_beams, dim=0)

    # ------------- helper -------------
    @staticmethod
    def _expand_kv(kv, beam):
        return tuple(
            tuple(layer.repeat_interleave(beam, dim=0) for layer in tup)
            for tup in kv
        )

    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:

        # return scores
        cur_len = input_ids.shape[1]

        Beta = 0.2 + cur_len/420 * 0.3
        Beta = 0.5

        # 第 0 步：直接返回缓存
        if cur_len == 0:
            return (1+Beta) * self.init_logits_main - Beta * self.init_logits_contra

        # 第 ≥1 步：走 KV-Cache
        B_cur = input_ids.shape[0]
        # print(cur_len)
        text_mask = torch.ones(B_cur, cur_len, dtype=torch.long,
                               device=input_ids.device)
        full_mask_main   = torch.cat([self.mask_main, text_mask], dim=1)
        full_mask_contra = torch.cat([self.mask_contra, text_mask], dim=1)

        with torch.no_grad():
            out_main = self.decoder(
                input_ids=input_ids[:, -1:],          # [B*beam, 1]
                past_key_values=self.past_main,
                attention_mask=full_mask_main,
                use_cache=True, return_dict=True)
            logits_main = out_main.logits[:, -1, :]
            self.past_main = out_main.past_key_values

            out_contra = self.decoder(
                input_ids=input_ids[:, -1:],
                past_key_values=self.past_contra,
                attention_mask=full_mask_contra,
                use_cache=True, return_dict=True)
            logits_contra = out_contra.logits[:, -1, :]
            self.past_contra = out_contra.past_key_values

        return (1+Beta) * logits_main - Beta * logits_contra

class BeamableRealtimeContrastProcessor_(LogitsProcessor):
    """
    Real-time contrastive decoding for **any num_beams**.
    用法和 generate 完全一致，只需把 num_beams 设为 >1 即可。
    """

    def __init__(
        self,
        decoder,
        img_embeds,        # [B, L_img, d]
        atts_img,          # [B, L_img]
        img_embeds_contra, # [B, L_img, d]
        atts_img_contra,   # [B, L_img]
        beta=0.5,
        num_beams=1
    ):
        super().__init__()
        self.decoder = decoder
        self.beta = beta
        self.num_beams = num_beams
        B, L_img, _ = img_embeds.shape
        device = img_embeds.device

        # 2. 按 beam 份数复制
        def expand(kv):
            # kv: tuple(tuple(Tensor))
            # 每个 Tensor shape: [B, n_heads, seq_len, d_head]
            return tuple(
                tuple(layer.expand(B * num_beams, -1, -1, -1) for layer in tup)
                for tup in kv
            )
        
        with torch.no_grad():
            out_main = decoder(
                inputs_embeds=img_embeds,
                attention_mask=atts_img,
                use_cache=True,
                return_dict=True
            )
            self.past_main = expand(out_main.past_key_values)
            # 缓存主图前缀最后一个 token 的 logits
            self.init_logits_main = out_main.logits[:, -1, :]   # [B, V]

            out_contra = decoder(
                inputs_embeds=img_embeds_contra,
                attention_mask=atts_img_contra,
                use_cache=True,
                return_dict=True
            )
            self.past_contra = expand(out_contra.past_key_values)
            self.init_logits_contra = out_contra.logits[:, -1, :]  # [B, V]




        # mask 也要复制
        self.mask_main   = atts_img.repeat_interleave(num_beams, dim=0)
        self.mask_contra = atts_img_contra.repeat_interleave(num_beams, dim=0)

    # def __call__(self, input_ids, scores, **kwargs):

    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[1]

        # 第 0 步：直接用 init 缓存
        if cur_len == 0:
            B_cur = input_ids.shape[0]
            logits_main = self.init_logits_main.repeat_interleave(
                B_cur // self.init_logits_main.size(0), dim=0)
            logits_contra = self.init_logits_contra.repeat_interleave(
                B_cur // self.init_logits_contra.size(0), dim=0)
            return logits_main - self.beta * logits_contra

        # 第 ≥1 步：正常走 KV-Cache
        device = input_ids.device
        B_cur = input_ids.shape[0]

        text_mask = torch.ones(B_cur, cur_len, dtype=torch.long, device=device)
        full_mask_main   = torch.cat([self.mask_main[:B_cur],   text_mask], dim=1)
        full_mask_contra = torch.cat([self.mask_contra[:B_cur], text_mask], dim=1)

        with torch.no_grad():
            out_main = self.decoder(
                input_ids=input_ids[:, -1:],
                past_key_values=self.past_main,
                attention_mask=full_mask_main,
                use_cache=True,
                return_dict=True
            )
            logits_main = out_main.logits[:, -1, :]
            self.past_main = out_main.past_key_values

            out_contra = self.decoder(
                input_ids=input_ids[:, -1:],
                past_key_values=self.past_contra,
                attention_mask=full_mask_contra,
                use_cache=True,
                return_dict=True
            )
            logits_contra = out_contra.logits[:, -1, :]
            self.past_contra = out_contra.past_key_values

        return logits_main - self.beta * logits_contra
    
    def __call____(self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor) -> torch.FloatTensor:
        device = input_ids.device

        # print(input_ids.shape)
        cur_len = input_ids.shape[1]          # 已生成 token 数量
        B_cur = input_ids.shape[0]            # B * beam（可能已 reorder）
        beam_indices = None
        # 1. 用 beam_indices 重排 KV-Cache
        if beam_indices is not None:
            self._reorder_cache(beam_indices)
        

        # 第 0 步：直接返回缓存值
        if cur_len == 0:
            B_cur = input_ids.shape[0]         # B * beam
            # 把缓存 logits 按 beam 复制
            logits_main   = self.init_logits_main.repeat_interleave(
                                 B_cur // self.init_logits_main.size(0), dim=0)
            logits_contra = self.init_logits_contra.repeat_interleave(
                                 B_cur // self.init_logits_contra.size(0), dim=0)
            return logits_main - self.beta * logits_contra

        # 第 ≥1 步：正常走 KV-Cache
        device = input_ids.device
        B_cur = input_ids.shape[0]

        if beam_indices is not None:
            self._reorder_cache(beam_indices)

        text_mask = torch.ones(B_cur, cur_len, dtype=torch.long, device=device)
        full_mask_main   = torch.cat([self.mask_main[:B_cur],   text_mask], dim=1)
        full_mask_contra = torch.cat([self.mask_contra[:B_cur], text_mask], dim=1)

        with torch.no_grad():
            out_main = self.decoder(
                input_ids=input_ids[:, -1:],   # [B*beam, 1]
                past_key_values=self.past_main,
                attention_mask=full_mask_main,
                use_cache=True,
                return_dict=True
            )
            logits_main = out_main.logits[:, -1, :]
            self.past_main = out_main.past_key_values

            out_contra = self.decoder(
                input_ids=input_ids[:, -1:],
                past_key_values=self.past_contra,
                attention_mask=full_mask_contra,
                use_cache=True,
                return_dict=True
            )
            logits_contra = out_contra.logits[:, -1, :]
            self.past_contra = out_contra.past_key_values

        return logits_main - self.beta * logits_contra

        # return logits_main - self.beta * logits_contra
    # ----------------- helper -----------------
    def _reorder_cache(self, beam_indices):
        """
        beam_indices: [B * beam]  long tensor
        """
        def reorder_single(kv):
            # kv: tuple(tuple(Tensor))
            return tuple(
                tuple(layer.index_select(0, beam_indices) for layer in tup)
                for tup in kv
            )
        self.past_main   = reorder_single(self.past_main)
        self.past_contra = reorder_single(self.past_contra)
        self.mask_main   = self.mask_main.index_select(0, beam_indices)
        self.mask_contra = self.mask_contra.index_select(0, beam_indices)