import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, BertTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertConfig, BertModel
from transformers import LlamaForCausalLM, LlamaTokenizer

import math
import torch.nn.functional as F


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

import re
from .patch_selection import select_features_network_variable_k_gpu,DefaultSelector_2, select_features_mmr_variable_k_gpu, DefaultSelector, select_features_network_mmr_variable_k_gpu_purefeature_r

from .online_utils import *
from .utils_GRPO import reward_think

from paths_config import (
    BIOMEDVLP_BERT,
    CTRG_MODEL_PATH,
    LLM_DECODER_PATH,
    MED_CONFIG_BLIP,
)
# v14 with 2 layer cross modal path selection
# v15 with feature level retrive enhancement
# v19 with more token

# v20 with high image dimension

# v30 with feature pre extracted

import json
import csv
import re

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


class M3AETransformerSS_3D_lmae_rg_v38_llm_v3(pl.LightningModule):
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
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-2-7b-chat-hf"

        
        med_config = MED_CONFIG_BLIP
        med_config = BertConfig.from_json_file(med_config)
        # tokenizer 改一下


        path = LLM_DECODER_PATH
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/llava_med"
        #path = "/jizhicfs/datalh/dataset/qwen/qwen7b"
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/Qwen3-4B"
 

        # self.text_decoder = LlamaForCausalLM.from_pretrained(
        self.text_decoder = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                # load_in_8bit=True,
                device_map="cuda"
            )
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-3.2-3B"
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = 0
        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size


        self.embed_tokens = self.text_decoder.get_input_embeddings()

        if True:
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

        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        
        # == End  : 1. Build Models ==

        # == Begin: 1.5 
        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        

        # self.crossAttentionLayer = CrossAttentionLayer(config["hidden_size"], 4)
        # self.crossAttentionLayer = QFormer(config["hidden_size"], 4)

        # self.selector = DefaultSelector(768)
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
        

        #self.p_enc_3d = PositionalEncodingPermute3D(config["hidden_size"]) 

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
    
    def infer_image_(self, expert, query, ifea, vision_extract_layer, lambda_mmr = 1, M = None, k_cur = 20):

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

            if M is None:
                selected_patch, details = select_features_network_mmr_variable_k_gpu_purefeature_r(y, attention_map, query, training = False, selector_model = self.selector, return_details = True, k_cur = k_cur)
            else:
                selected_patch, details = select_features_network_mmr_variable_k_gpu_purefeature_r(y, attention_map, query, training = False, M = M, selector_model = self.selector, return_details = True, k_cur = k_cur)
            
            final_f = []

            for iii in range(b):
                final_f.append(torch.cat(selected_patch[iii], 0).unsqueeze(0))
            final_f = torch.cat(final_f, 0)
            selected = torch.cat([final_f, x], 1)
        # print(selected.shape)
        return selected

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
            cprompt = self.prompt

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

    def prompt_wrap_2(self, img_embeds, atts_img, prior = None, cls_res = None):

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
            cprompt = self.prompt

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
        # print(batch["impression_seg_text"])
        bs = img.shape[0]
        # print(batch["all_encodings_i"], batch["all_encodings_i"].float().mean(1))
        if np.random.rand() > -1 or True:
            local_text_i, local_text_word_i = self.infer_text(self.text_model, self.contrastive_proj_text, batch["all_encodings_i"], batch["all_maps_i"], device)
        else:
            local_text_i, local_text_word_i = self.infer_text(self.text_model, self.contrastive_proj_text, batch["all_encodings_i_rej"], batch["all_maps_i_rej"], device)

        local_text_word_i = local_text_word_i.reshape(bs, 10, local_text_word_i.shape[-1])
        D = local_text_word_i # torch.where(impression_idx.squeeze(1).unsqueeze(-1) == 0, local_text_word_i, self.expert_text.weight.unsqueeze(0).repeat(local_text_word_i.shape[0], 1, 1))
        # print(D.shape)
        plan = np.random.rand()
        # plan = 1


        # get the imagean embedding
        # uni_modal_image_feats = img.cuda().squeeze(1)
        # == Training mode selector ==
        #   "full"   -> train the full report-generation model: online expert
        #               patch extraction from raw image features, supervised with
        #               the "<think> impression </think>. <answer> finding </answer>"
        #               target (abnormality reasoning + full report).
        #   "abnorm" -> train the abnormality-extraction model only: use the
        #               precomputed selected patches and supervise with the
        #               impression (abnormality) text alone.
        train_mode = self.config.get("train_mode", "full")
        if train_mode == "full":
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
            
        elif train_mode == "abnorm":
            selected_patch = img.squeeze(1)
            
            selected_patch = self.multi_modal_vision_proj_forllm(selected_patch)

            image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                    device=device)
            
            img_embeds, atts_img = self.prompt_wrap_2(selected_patch, image_masks, batch["prior"])
        

            impression = batch["impression"]
            # print(impression)

            text_input = impression
        else:
            raise ValueError(
                f"Unknown train_mode={train_mode!r}; expected 'full' or 'abnorm'."
            )

        to_regress_tokens = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=450,
            add_special_tokens=False
        ).to(device)


        
        # print(batch["rg_encoding"])
        # print(text_labels.input_ids.shape)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        #print(to_regress_embeds.device, "xxxxxxxxxxxxxxxxxxxxx")

        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
        # print(self.embed_tokens)
        # self.text_decoder = self.text_decoder.to(device)
        # print(device)
        # print(image_masks.shape, attention_map.shape) # 1,45   //  1,96,9
        # selected_patch[:, 5:-1, :] = selected_patch[:, -1:, :]
        # print(torch.isnan(inputs_embeds).any())
        decoder_output = self.text_decoder(inputs_embeds=inputs_embeds.to(device),
                                           attention_mask=attention_mask.to(device),
                                           return_dict=True,
                                           labels=targets.to(device),
                                          )
        #print(decoder_output.shape)
        # sprint(decoder_output.logits.shape) 1 524 32000
        #print(decoder_output[:, :10, :])
        #exit(0)

        # == Begin: == Output Multi-Modal Features ==
        if not self.training:
            # self.text_decoder.config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.text_decoder.config.eos_token_id

            outputs = self.text_decoder.generate(
                inputs_embeds=img_embeds,
                attention_mask = atts_img,
                num_beams=1,
                do_sample=False,
                min_new_tokens=20,
                max_new_tokens=450,
                return_dict_in_generate=True, 
                pad_token_id= self.tokenizer.eos_token_id # 128001  # 或者使用其他合适的pad_token_id
                # output_attentions=True,
            )

            # print(next(self.text_decoder.parameters()).dtype)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            # print(outputs.keys())
            # batch["impression"]

            captions = []
            for output in outputs.sequences:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[:])

            if False:
                store_path = self.ckpath + "_ori.txt"
                text_file = open(store_path, "a+")
                text_file.writelines("****" + "\n")
                text_file.writelines(captions[0] + "\n")

                thinks = [] # only think
                for an in captions:
                    match = re.search(r'<answer>(.*)', an, re.DOTALL)

                    # 提取think
                    pattern = r'<think>(.*?)<\/think>'

                    # 使用re.search()提取第一个匹配的内容
                    match = re.search(pattern, an)

                    # 检查是否有匹配，并打印提取的内容
                    if match:
                        # print(match.group(1))
                        thinks.append(match.group(1))
                    else:
                        thinks.append("not given")

                # thinks = impression
                # print(D.shape)
                f1s, captions = reward_think(thinks, batch["data_name"], batch["prior"], obver_label.cpu())
                # print(f1s)
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
            "captions": captions,
            "data_name": batch["data_name"],
            "obver_label": obver_label,
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
            ret.update(objectives.compute_rg_F1_online(self, batch))

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