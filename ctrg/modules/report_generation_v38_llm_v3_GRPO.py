import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, BertTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertConfig, BertModel
from transformers import LlamaForCausalLM, LlamaTokenizer

import math
import torch.nn.functional as F
from pathlib import Path
import csv

from ctrg.modules import objectives, m3ae_utils
from ctrg.modules import prediction_heads
from ctrg.modules.language_encoders.bert_model import BertCrossLayer
from ctrg.modules.m3ae_utils import init_weights


from ctrg.modules.RadFM.vit_3d import ViT

from ctrg.modules.models.med import BertConfig, BertLMHeadModel
from ctrg.modules.models.vis_utils import save_attention_maps, compute_combined_attention, save_average_cross_attention, compute_last_token_attention_proportion

import numpy as np
#######################################
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

from peft import get_peft_model, LoraConfig, TaskType

from positional_encodings.torch_encodings import PositionalEncodingPermute3D
from scipy.special import expit
# from transformer_maskgit import CTViT
from ctvit import CTViT
from classifier import RadBertClassifier
from .utils_GRPO import *
from .online_utils import *
from transformers import GenerationConfig


import re

from .utils_GRPO import seg_text
from .patch_selection import select_features_network_variable_k_gpu, select_features_mmr_variable_k_gpu, DefaultSelector_2, DefaultSelector, select_features_network_mmr_variable_k_gpu_purefeature_r

from paths_config import (
    BIOMEDVLP_BERT,
    CTRG_MODEL_PATH,
    LLM_DECODER_PATH,
    MED_CONFIG_BLIP,
    GRPO_REF_SERVER_URL,
)

from .online_utils import *

# v14 with 2 layer cross modal path selection
# v15 with feature level retrive enhancement
# v19 with more token

# v20 with high image dimension

# v30 with feature pre extracted

class SimpleMLP(nn.Module):
    def __init__(self, indim, outdim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=indim, out_features=indim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=indim, out_features=outdim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Define linear projection layers for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multihead attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, add_residual=False):
        # Apply linear projections
        query = self.query_proj(query).permute(1, 0, 2)
        key = self.key_proj(key).permute(1, 0, 2)
        value = self.value_proj(value).permute(1, 0, 2)
        # print(query.shape, key.shape, value.shape)
        # Compute cross-attention
        attn_output, _ = self.multihead_attn(query, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output = attn_output + query
        
        # Apply normalization and dropout
        attn_output = self.norm(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output.permute(1, 0, 2)

def csv_to_dict(filepath):
    """
    读取CSV文件，第一列作为key，第三列作为value，第一行也是数据
    """
    result = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader, start=1):
                if len(row) < 3:
                    print(f"⚠️ 警告: 第 {idx} 行列数不足3，已跳过: {row}")
                    continue
                key = row[0].strip()
                value = row[2].strip()
                if key:  # 跳过空key
                    result[key] = value
    except FileNotFoundError:
        print(f"❌ 错误: 文件 '{filepath}' 不存在")
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
    
    return result

class M3AETransformerSS_3D_lmae_rg_v38_llm_v3_GRPO(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.momentum = 0.99
        self.queue_size = 512
        self.temp = 0.07
        self.aligned_length = 512
        self.selected_patch_n = config["selected_patch"]
        print(self.selected_patch_n)
        self.num_iters = 1

        self.is_clip = ('swin' not in config['vit'])
        if 'roberta' in config['tokenizer'] and False:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        elif 'bert' in config['tokenizer'] or True:

            textmodelpath = CTRG_MODEL_PATH
            bert_config = AutoConfig.from_pretrained(textmodelpath)
        else:
            raise ValueError

        # == Begin: 1. Build Models ==
        
        ####for the ce metric cal

        # == vision encoder ==

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        textmodelpath = BIOMEDVLP_BERT
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        self.text_model = BertModel.from_pretrained(textmodelpath)
        self.tokenizer_text_model = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)


        # self.hparams.config["vocab_size"] = self.tokenizer.vocab_size

        med_config = MED_CONFIG_BLIP
        med_config = BertConfig.from_json_file(med_config)
        # tokenizer 改一下


        path = LLM_DECODER_PATH
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-3.2-1B"
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/llava_med"
        #path = "/jizhicfs/datalh/dataset/qwen/qwen7b"
 

        # self.text_decoder = LlamaForCausalLM.from_pretrained(
        self.text_decoder = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                # load_in_8bit=True,
                device_map="cuda"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = 0
        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size

        self.embed_tokens = self.text_decoder.get_input_embeddings()

        if True:
            lora_config = LoraConfig(
                r=64, # 64
                lora_alpha=16,
                target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj",],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
                #modules_to_save=modules_to_save  # This argument serves for adding new tokens.
            )

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128, lora_alpha=16, lora_dropout=0.05,target_modules=["W_pack", "o_proj"]
            )

            self.text_decoder = get_peft_model(self.text_decoder, peft_config)
            self.text_decoder.print_trainable_parameters()
            print('Loading LLAMA LoRA Done') 
        
        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size
        
        #########################################

        resolution_after = config['image_size']

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        
        # == End  : 1. Build Models ==

        # == Begin: 1.5 
        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        
        self.text_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(1)])

        # self.crossAttentionLayer = CrossAttentionLayer(config["hidden_size"], 4)
        # self.crossAttentionLayer = QFormer(config["hidden_size"], 4)
        self.selector = DefaultSelector_2(768)
        
        self.contrastive_proj_image = nn.Linear(config["hidden_size"], self.aligned_length)
        self.contrastive_proj_image2 = nn.Linear(768, config["hidden_size"])
        self.contrastive_proj_text = nn.Linear(config["hidden_size"], self.aligned_length)
        self.obver_norm_class = nn.Linear(768, 1)

        self.multi_modal_vision_proj_forllm = SimpleMLP(config['hidden_size'], 3072) # 2560 for 4b, 2048 for 1.8b, 1024for.5b, 1536 for 2 1.5b 896
        self.multi_modal_vision_proj_forllm.apply(init_weights)

        # class token
        
        self.expert_image = nn.Embedding(10, config["hidden_size"])
        self.expert_image.apply(init_weights)

        self.expert_text = nn.Embedding(10, config["hidden_size"])
        self.expert_text.apply(init_weights)
        
        self.image_position_embeddings = nn.Embedding(96, config["hidden_size"])

        self.p_enc_3d = PositionalEncodingPermute3D(config["hidden_size"]) 
        #self.p_enc_3d = PositionalEncodingPermute3D(config["hidden_size"]) 

        self.buffer = []
        # == End

        # waiting

        m3ae_utils.set_metrics(self)

        self.entropy_record = torch.zeros([1,380])
        self.entropy_sum = torch.ones([1,380])


    def infer_image(self, vision_encoder, vision_extract_layer, vision_contrastive_proj, contrastive_proj_image2, expert, img, u=0):
        uni_modal_image_feats = vision_encoder(img.unsqueeze(1), return_encoded_tokens=True)  # _, c, h, w, d
        b, x1,x2,x3, d = uni_modal_image_feats.shape
        uni_modal_image_feats = uni_modal_image_feats.reshape(b, -1, d)
        b, n, d = uni_modal_image_feats.shape

        uni_modal_image_feats = contrastive_proj_image2(uni_modal_image_feats)

        # == Begin: extract key image feature
        x, y = expert.weight.unsqueeze(0), uni_modal_image_feats
        for layer_idx, (extract_layer) in enumerate(vision_extract_layer):
            x1 = extract_layer(x, y, output_attentions = True)
            y1 = extract_layer(y, x, output_attentions = True)

            x, y = x1[0], y1[0]
            x_attention = x1[1:]
            y_attention = y1[1:]

        attention_map = x_attention[1].mean(1) #.softmax(1)

        _, selected_index = torch.sort(attention_map, 2)


        selected_patch = []
        for ii in range(x.shape[0]):
            selected_patch.append(torch.cat([torch.cat([x[ii:ii+1, expert_i:expert_i + 1, :], y[ii:ii+1, selected_index[ii, expert_i, -self.selected_patch_n:], :]], 1) for expert_i in range(10)], 1)) # use y or orginal uni_modal_image_feats
        selected_patch = torch.cat(selected_patch, 0)


        local_ = selected_patch

        local_image = vision_contrastive_proj(x).reshape(-1, self.aligned_length)

        # selected_patch = torch.cat([selected_patch, x], 1)
        selected_patch = x

        # selected_patch = uni_modal_image_feats * attention_map.unsqueeze(-1)
        if u==0:
            norm_class_res = self.obver_norm_class(x)[0, :8].reshape(1,8)
            return selected_patch, local_image, norm_class_res, x_attention[1].mean((1))
        return selected_patch, local_image, x_attention[1].mean((1))
    
    def infer_image_(self, expert, query, ifea, vision_extract_layer, lambda_mmr = 1):

        b, n, d = ifea.shape

        # == Begin: extract key image feature
        x, y = expert.weight.unsqueeze(0), ifea
        with torch.no_grad():
            for layer_idx, (extract_layer) in enumerate(vision_extract_layer):
                extract_layer.eval()
                x1 = extract_layer(x, y, output_attentions = True)
                y1 = extract_layer(y, x, output_attentions = True)

                x, y = x1[0], y1[0]
                x_attention = x1[1:]
                y_attention = y1[1:]
        
            attention_map = x_attention[1].mean(1) # .softmax(1)

            selected_patch, details = select_features_network_mmr_variable_k_gpu_purefeature_r(y, attention_map, query, training = False, selector_model = self.selector, return_details = True)
            
            final_f = []

            for iii in range(b):
                final_f.append(torch.cat(selected_patch[iii], 0).unsqueeze(0))
            final_f = torch.cat(final_f, 0)
            selected = torch.cat([final_f, x], 1)
        # print(selected.shape)
        return selected


    def infer_text(self, text_model, contrastive_proj, input_ids, maps, device):
        
        # print(text_seg_index)
        #print(index.shape)
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        maps = maps.reshape(-1, input_ids.shape[-1])
        # print(text_model.device, device)
        with torch.no_grad():
            
            local_text_ori = text_model(input_ids, attention_mask = maps)[0][:, 0:1, :] # .permute(1,0,2)
                # print(seg)
                # print(local_feature.shape)
        # print(local_text.shape)
        #local_text = torch.cat(local_text, 1)
        #local_text = local_text.permute(1,0,2)

        #print(local_text.shape, "hhhh"
        # print(local_text.shape, "fxxk")# 4*10
        # attention = attention_with_norm(local_text.permute(1,0,2), local_text.permute(1,0,2))
        # print(attention)
        
            local_text = contrastive_proj(local_text_ori).reshape(-1, self.aligned_length)
        return local_text, local_text_ori #local_text_ori
    
    def prompt_wrap(self, img_embeds, atts_img, prior = None, cls_res = None):

        if cls_res is not None:
            abnorm = [score.cpu().numpy() for score in cls_res[0].detach().sigmoid()]
            struct_list = ["Thoracic strctures", "Rib", "Lung", "Heart", "Pleural", "Liver", "Kidney", "Thyroid"]
            abnorm_str = ""
            for iii, struct in enumerate(struct_list):
                abnorm_str += struct + ": " + str(abnorm[iii]) + "; "
            cprompt  = self.prompt + " The abnorm probability of each of eight structure are: " + abnorm_str
        elif prior is not None:
            cprompt = """Generate a comprehensive and detailed diagnosis report for those features selected from a chest CT image based on image representation, clinical info and last report abnormality (if available). The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
                        The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
        else:
            cprompt = """Generate a comprehensive and detailed diagnosis report for those features selected from a chest CT image based on image representation. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
                        The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


        # print(cprompt)
        prompt=f'Human: <Img><ImageHere></Img> {cprompt} \nAssistant:'

        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)


        if prior is not None:
            # print(prior)
            prior_token = self.tokenizer(prior, 
                                        return_tensors="pt", 
                                        add_special_tokens=False, 
                                        max_length=128,  # 设置最大长度为128
                                        padding='max_length',  # 使用最大长度进行填充
                                        truncation=True).to(img_embeds.device)
            
            prior_embeds = self.embed_tokens(prior_token.input_ids)
            # print(prior_embeds.shape, p_before_embeds.shape, img_embeds.shape)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, prior_embeds, p_after_embeds], dim=1)
        else:
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        # print(p_before_embeds.shape, img_embeds.shape, p_after_embeds.shape)

        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def prompt_wrap_2(self, img_embeds, atts_img, prior = None, cls_res = None):
        if cls_res is not None:
            abnorm = [score.cpu().numpy() for score in cls_res[0].detach().sigmoid()]
            struct_list = ["Thoracic strctures", "Rib", "Lung", "Heart", "Pleural", "Liver", "Kidney", "Thyroid"]
            abnorm_str = ""
            for iii, struct in enumerate(struct_list):
                abnorm_str += struct + ": " + str(abnorm[iii]) + "; "
            cprompt  = self.prompt + " The abnorm probability of each of eight structure are: " + abnorm_str
        elif prior is not None:
            cprompt = """Generate a comprehensive and detailed diagnosis report for those features queried from a chest CT image based on image representation, clinical info and last report abnormality (if available). The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
                        The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
        else:
            cprompt = """Generate a comprehensive and detailed diagnosis report for those features selected from a chest CT image based on image representation. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
                        The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""


        # print(cprompt)
        prompt=f'Human: <Img><ImageHere></Img> {cprompt} \nAssistant:'

        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        

        # print(prior_embeds.shape)

        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)


        if prior is not None:
            # print(prior)
            prior_token = self.tokenizer(prior, 
                                        return_tensors="pt", 
                                        add_special_tokens=False, 
                                        max_length=128,  # 设置最大长度为128
                                        padding='max_length',  # 使用最大长度进行填充
                                        truncation=True).to(img_embeds.device)
            
            prior_embeds = self.embed_tokens(prior_token.input_ids)
            # print(prior_embeds.shape, p_before_embeds.shape, img_embeds.shape)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, prior_embeds, p_after_embeds], dim=1)
        else:
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        # print(p_before_embeds.shape, img_embeds.shape, p_after_embeds.shape)

        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def gen_grpo_buffer(self, img_embeds, atts_img, obver_label, text_input, text_output, device, selected_patch, image_masks, batch_prior, dataname, plan):
        self.buffer = []

        self.num_pre_Q = 5
        generation_config = GenerationConfig(
            max_new_tokens=220,
            do_sample=True, temperature=0.9, 
            num_return_sequences=self.num_pre_Q,
            pad_token_id= self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            top_p = 0.95
            # num_beams=10,          # 或 16、32 等你想要的宽度
            # early_stopping=True,  # 可选，让 beam 提前停止
        )
        # generate multiple results
        # with torch.inference_mode():
        completion_ids = self.text_decoder.generate(inputs_embeds=img_embeds, attention_mask = atts_img, generation_config = generation_config).clone()
        answers = [self.tokenizer.decode(x, skip_special_tokens=True).replace('<|endoftext|>', '') for x in completion_ids]
        print(answers[0])
        answers_ = [] # with out think
        thinks = answers # only think
        
        # print(answers[0])        

        if plan < 0.5:

            f1s, reports = reward_think(thinks, dataname, batch_prior, obver_label.cpu(),use_green = False)

        print(f1s, reports[0])
        # print(f1s)

        # get rewards
        rewards = []
        for ii, l in enumerate(answers):
            blue = reward_blue(reports[ii], obver_label.cpu(), text_output[0])

            if plan < 0.5:
                rewards.append(f1s[ii]  + blue)
            else:
                rewards.append(F1 + blue + reward_format(l))
                
        rewards = torch.tensor(rewards, dtype = torch.float32)

        print(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    
        # get the ref logits from server
        ref_server = GRPO_REF_SERVER_URL
        prompt_length = img_embeds.shape[1]
        completion_embedding = self.embed_tokens(completion_ids)
        prompt_embedding = img_embeds.repeat(1, self.num_pre_Q, 1).view(-1, prompt_length, img_embeds.shape[2])
        total_embedding = torch.cat([prompt_embedding, completion_embedding], dim = 1)

        all_mask = torch.cat([atts_img.repeat(self.num_pre_Q, 1), torch.ones(completion_embedding.shape[:2],).to(device)], dim = 1)

        logits = self.text_decoder(inputs_embeds = total_embedding.to(device), attention_mask = all_mask).logits[:, prompt_length:-1, :].clone()
        gen_logps = get_per_token_logps(logits, completion_ids[:, 1:].clone())
        

        data = [json.dumps({"plen": prompt_length}).encode(), tensor_to_bytes(total_embedding), tensor_to_bytes(rewards), tensor_to_bytes(completion_ids)]

        data.append(tensor_to_bytes(gen_logps.cpu()))
        xdata = make_bytes_list(data)
        response = requests.post(f"{ref_server}/upload", data=xdata)
        request_id = response.content.decode('utf-8')

        r = requests.get(f"{ref_server}/get/{request_id}").content
        while r == b'empty':
            r = requests.get(f"{ref_server}/get/{request_id}").content

        dd = bytes_list_to_list(r)
        data = json.loads(dd[0]) 
        
        data['rewards'] = bytes_to_tensor(dd[1])
        data['refs'] = bytes_to_tensor(dd[2])
        data['gen_logps'] = bytes_to_tensor(dd[3])

        data['inputs'] = total_embedding
        data['completion_ids'] = completion_ids
        data['attn_mask'] = all_mask

        self.buffer.append(data)

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            unimodal=False
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            img = batch[img_key]
        #do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids"]
        #text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]
        text_seg_index = batch[f"seg_index"]
        # text_ori = batch[f"text"]
        obver_label = batch[f"obver_label"]
        #print(text_ori)
        # print(text_seg_index)
        device = text_ids.device
        ####################################################

        # get the impression embedding
        impression_idx = batch["impression_seg"]
        bs = img.shape[0]

        local_text_i, local_text_word_i = self.infer_text(self.text_model, self.contrastive_proj_text, batch["all_encodings_i"], batch["all_maps_i"], device)
        local_text_word_i = local_text_word_i.reshape(bs, 10, local_text_word_i.shape[-1])
        D = local_text_word_i # torch.where(impression_idx.squeeze(1).unsqueeze(-1) == 0, local_text_word_i, self.expert_text.weight.unsqueeze(0).repeat(local_text_word_i.shape[0], 1, 1))

        plan = np.random.rand()
        plan = 0
        # get the image embedding
        if plan > 0.5: #  and self.training:
            uni_modal_image_feats = batch["all_image"].cuda().squeeze(1)
            with torch.no_grad():
                # _, selected_patch = self.crossAttentionLayer(D, uni_modal_image_feats, add_residual = False)
                # print(selected_patch.shape)
                local_image = self.infer_image_(self.expert_image, D, uni_modal_image_feats, self.vision_extract_layer)
                
            selected_patch = self.multi_modal_vision_proj_forllm(local_image)

            image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                    device=device)
            
            img_embeds, atts_img = self.prompt_wrap(selected_patch, image_masks, batch["prior"])
        

            impression = batch["impression"]
        
            finding = batch["text_ori"]


            text_input = []
            for idx, imp in enumerate(impression):
                text_input.append("<think> " + imp + "</think>. " + "<answer>" + finding[idx] + "</answer>")
            
        else:
            selected_patch = img.squeeze(1)
            
            selected_patch = self.multi_modal_vision_proj_forllm(selected_patch)

            image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                    device=device)
            
            img_embeds, atts_img = self.prompt_wrap_2(selected_patch, image_masks, batch["prior"])
        

            impression = batch["impression"]
            finding = batch["text_ori"]

            text_input = impression

            #@ text_output = batch["text_ori"]

            text_output = []
            for idx, imp in enumerate(impression):
                text_output.append("<think> " + imp + "</think>. " + "<answer>" + finding[idx] + "</answer>")
        # print(plan, text_input)

        # text_input = batch["text_ori"]
        to_regress_tokens = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=220,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )

        targets = torch.cat([empty_targets[:,:1], targets[:,:1]], dim=1)

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([img_embeds[:,:1], to_regress_embeds[:,:1]], dim=1)
        attention_mask = torch.cat([atts_img[:,:1], to_regress_tokens.attention_mask[:,:1]], dim=1)

        decoder_output = self.text_decoder(inputs_embeds=inputs_embeds.to(device),
                                           attention_mask=attention_mask.to(device),
                                           return_dict=True,
                                           labels=targets.to(device),
                                          )

        if not self.training:
            grpo_loss = 0.0
        else:
            if self.global_step % self.num_iters == 0 or len(self.buffer) == 0:
                with torch.no_grad():
                    self.gen_grpo_buffer(img_embeds, atts_img, obver_label, text_input, text_output, device, selected_patch, image_masks , batch["prior"], batch["data_name"], plan)
            
            data = self.buffer[-1]

            logits = self.text_decoder(inputs_embeds = data['inputs'].to(device), attention_mask = data['attn_mask'].to(device), labels = torch.cat([empty_targets.repeat(self.num_pre_Q, 1), data['completion_ids'].to(device)], dim=1)).logits

            grpo_loss = GRPO_step(data, device, self.tokenizer, logits)

            print(grpo_loss)

        # == Begin: == Output Multi-Modal Features ==
        if not self.training:
            # self.text_decoder.config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.text_decoder.config.eos_token_id

            outputs = self.text_decoder.generate(
                inputs_embeds=img_embeds,
                attention_mask = atts_img,
                num_beams=1,
                do_sample=False,
                min_new_tokens=30,
                max_new_tokens=220,
                return_dict_in_generate=True, 
                # repetition_penalty=1.05,
                # top_p = 0.95,
                pad_token_id= self.tokenizer.eos_token_id # 128001  # 或者使用其他合适的pad_token_id
                # output_attentions=True,
            )

            # print(outputs.keys())
            # batch["impression"]

            captions = []
            for output in outputs.sequences:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[:])

            # captions = []
            # results_ = csv_to_dict("/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/results/task_pretrain_m3ae-seed0-from_/version_826/checkpoints/epoch=0-step=2400.ckpt_directconnecttoversion764_b1_v2_ori.csv")
            # results_ = csv_to_dict("/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/results/task_pretrain_m3ae-seed0-from_/version_835/checkpoints/epoch=0-step=3000.ckpt_directconnecttoversion764_b1_v2_ori.csv")
            # for l in batch["data_name"]:
            #     captions.append(results_[l])
            if True:

                store_path = getattr(self, f"ckpath") + "_ori.csv"
                # 把 Tensor 整体搬到 numpy，避免循环里频繁 .cpu()
                obver_np = batch["obver_label"].cpu().numpy()
                
                # 安全写入
                from pathlib import Path
                import csv

                with Path(store_path).open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # 同时迭代三列，长度不一致时按最短的来
                    for name, label, cap in zip(batch["data_name"], obver_np, captions):
                        # print(name, label, cap)
                        writer.writerow([name, label, cap])

                thinks = captions

                # print(D.shape)
                
                f1s, captions = reward_think(thinks, batch["data_name"], batch["prior"], obver_label.cpu(), use_green = False)
                if False:
                    D = seg_text(thinks[0], self.text_model, self.tokenizer_text_model).reshape(bs, 10, local_text_word_i.shape[-1])

                    uni_modal_image_feats = batch["all_image"].cuda().squeeze(1)
                    with torch.no_grad():
                        uni_modal_image_feats = batch["all_image"].cuda().squeeze(1)
            
                        # _, selected_patch = self.crossAttentionLayer(D, uni_modal_image_feats, add_residual = False)
                        # print(selected_patch.shape)
                        selected_patch = self.infer_image_(self.expert_image, D, uni_modal_image_feats, self.vision_extract_layer)
                        # print(selected_patch.shape)

                    selected_patch = self.multi_modal_vision_proj_forllm(selected_patch)

                    image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                            device=device)
                    
                    img_embeds, atts_img = self.prompt_wrap(selected_patch, image_masks, batch["prior"])
                

                    impression = batch["impression"]
                
                    finding = batch["text_ori"]


                    text_input = []
                    for idx, imp in enumerate(impression):
                        text_input.append("<think> " + imp + "</think>. " + "<answer>" + finding[idx] + "</answer>")
                    
                    outputs = self.text_decoder.generate(
                        inputs_embeds=img_embeds,
                        attention_mask = atts_img,
                        num_beams=1,
                        do_sample=False,
                        min_new_tokens=50,
                        max_new_tokens=420,
                        return_dict_in_generate=True, 
                        pad_token_id= self.tokenizer.eos_token_id # 128001  # 或者使用其他合适的pad_token_id
                        # output_attentions=True,
                    )

                    captions = []
                    for output in outputs.sequences:
                        caption = self.tokenizer.decode(output, skip_special_tokens=True)
                        captions.append(caption[:])
            

        else:
            captions = ["I am fxxking man."] * bs # for test 

        # print(text_ori)
        #print("xxxxxxxxxxxxxx")
        #print(captions)

        ret.update({
            "images": img,
            # "patched_images": self.patchify(img), # patch3D
            #"text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            #"extended_image_masks": extended_image_masks,
            #"extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": decoder_output,
            #"multi_modal_image_feats": multi_modal_image_feats,
            #"multi_modal_cls_feats": multi_modal_cls_feats,
            "decoder_output": decoder_output,
            "text_ori": text_input,
            "data_name": batch["data_name"],
            "captions": captions,
            "obver_label": obver_label,
            "grpo_loss": grpo_loss,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm2(self, batch))

        # Pre-Training: Masked Image Modeling
        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        # Pre-Training: Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Fine-Tuning: Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        # Fine-Tuning: Image-Text Classification
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))

        # Fine-Tuning: Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, test))
        
        if "rg" in self.current_tasks:
            ret.update(objectives.compute_rg_F1_online_grpo(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        

        output = self(batch)
        # print(self.hparams.config)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])
        # print(output, total_loss)
        return total_loss

    '''
    def training_step_end(self, batch_parts):

        print(batch_parts)

        return torch.mean(batch_parts)
    '''

    def on_training_epoch_end(self):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def on_test_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return m3ae_utils.set_schedule(self)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats, obverlabels):
        # gather keys before updating queue
        # print(image_feats.shape, text_feats.shape)
        for idx in range(obverlabels.shape[0]):
            obverlabel = obverlabels[idx]
            image_feat = image_feats[idx]
            text_feat = text_feats[idx]

            ttidx = []
            for iidx in range(obverlabel.shape[0]):
                if obverlabel[iidx]:
                    ttidx.append(iidx)
            #obverlabel = torch.cat([obverlabel, torch.zeros(1, device = obverlabel.device)], 0)
            batch_size = image_feat.shape[0]
            unnormsize = obverlabel.sum()
            image_feat = image_feat[ttidx]
            text_feat = text_feat[ttidx]

            ptr = int(self.queue_ptr)
            # assert self.queue_size % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            if ptr + unnormsize > self.queue_size:
                image_feat = image_feat[:self.queue_size-ptr]
                text_feat = text_feat[:self.queue_size-ptr]
            
            self.image_queue[:, ptr:ptr + unnormsize] = image_feat.T.detach()
            self.text_queue[:, ptr:ptr + unnormsize] = text_feat.T.detach()
            ptr = (ptr + unnormsize) % self.queue_size  # move pointer

            self.queue_ptr[0] = ptr 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    