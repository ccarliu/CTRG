import pytorch_lightning as pl
import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig
from transformers import RobertaConfig, RobertaModel, BertTokenizer, BertModel

import math
import torch.nn.functional as F

import numpy as np


from ctrg.modules import objectives, m3ae_utils
from ctrg.modules import prediction_heads
from ctrg.modules.language_encoders.bert_model import BertCrossLayer
from ctrg.modules.m3ae_utils import init_weights


from ctrg.modules.models.med import BertConfig, BertLMHeadModel
from ct_clip import CTCLIP

#######################################
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead

from positional_encodings.torch_encodings import PositionalEncodingPermute3D

from ctrg.modules.RadFM.vit_3d import ViT, ViT_expert, ViT_reg

from .patch_selection import select_features_network_variable_k_gpu, DefaultSelector_2, select_features_mmr_variable_k_gpu, select_features_network_mmr_variable_k_gpu_purefeature_2, DefaultSelector, select_features_network_mmr_variable_k_gpu_purefeature_r

# from transformer_maskgit import CTViT
from .m3ae_utils import feature_similarity_matrix
from ctvit import CTViT

from paths_config import (
    BIOMEDVLP_BERT,
    CT_CLIP_CKPT,
    CTRG_MODEL_PATH,
    TEXT_LATENT_NPZ,
)

# v14 with multi image per batch
# v15 with only itc
# v17 with change the text encoder
# v18, simple code
# v19: new patch selector back
# v21: v20 with a nagetive sample selective module to improve the quanity of nagetive pool.

# v24: with big resolution

# further version of v 24

def attention_with_norm(tensor1, tensor2):
    """
    计算两个形状为 (b, n, l) 的张量之间的注意力图，并在计算前进行层归一化。
    
    参数:
    - tensor1: 形状为 (b, n, l) 的张量
    - tensor2: 形状为 (b, n, l) 的张量
    
    返回:
    - attention: 形状为 (b, n, n) 的注意力图
    """
    b, n, l = tensor1.shape
    
    # 应用层归一化
    tensor1 = F.normalize(tensor1, p=2, dim=-1)
    tensor2 = F.normalize(tensor2, p=2, dim=-1)
    
    # 计算相似度矩阵
    similarity = torch.bmm(tensor1, tensor2.transpose(1, 2))  # 形状为 (b, n, n)
    
    # 计算注意力图
    attention = F.softmax(similarity, dim=-1)
    
    return attention

class QFormer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(QFormer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Define linear projection layers for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multihead attention layers for cross-attention
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, query, key, value, add_residual=True):
        # Apply linear projections
        query = self.query_proj(query).permute(1, 0, 2)
        key = self.key_proj(key).permute(1, 0, 2)
        value = self.value_proj(value).permute(1, 0, 2)
        
        # Compute first cross-attention
        attn_output1, _ = self.cross_attn1(query, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output1 = attn_output1 + query
        
        # Apply normalization and dropout
        attn_output1 = self.norm1(attn_output1)
        attn_output1 = self.dropout1(attn_output1)
        
        # Compute second cross-attention
        attn_output2, _ = self.cross_attn2(attn_output1, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output2 = attn_output2 + attn_output1
        
        # Apply normalization and dropout
        attn_output2 = self.norm2(attn_output2)
        attn_output2 = self.dropout2(attn_output2)
        
        return attn_output2.permute(1, 0, 2)

class M3AETransformerSS_3D_lmae_rg_pretrain_v29_Qwen_perorgan_postalign(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.momentum = 0.99
        self.queue_size = 10000
        self.temp = 0.07
        self.num_expert = 10
        self.aligned_length = 512

      
        self.bs = config['batch_size']
        mask_ori = (torch.ones(self.num_expert, self.num_expert) * 0.1).fill_diagonal_(1).unsqueeze(0)
        mask1 = mask_ori.repeat(self.bs*self.bs, 1, 1).reshape(self.bs, self.bs, self.num_expert, self.num_expert).permute(0,2,1,3).reshape(self.bs*self.num_expert, self.bs*self.num_expert)
        mask2 = mask_ori.repeat(self.bs*self.queue_size // self.num_expert, 1, 1).reshape(self.bs, self.queue_size // self.num_expert, self.num_expert, self.num_expert).permute(0,2,3,1).reshape(self.bs*self.num_expert, self.num_expert*self.queue_size // self.num_expert)
        
        
        self.imt_mask = torch.cat([mask1, mask2], 1)

        print(self.imt_mask.shape)


        loaded_data = np.load(TEXT_LATENT_NPZ)
        # print(loaded_data.shape)
        # 访问加载的数据
        loaded_names = loaded_data['names']
        self.test_text_features = F.normalize(torch.tensor(loaded_data['features']), dim = -1) # [300:800]) # F.normalize(torch.tensor(loaded_data['features'][300:800]), dim = -1)
        print(self.test_text_features.shape)
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
        textmodelpath = CTRG_MODEL_PATH
        config_m = AutoConfig.from_pretrained(textmodelpath)

        # == vision encoder ==


        self.vision_encoder = CTViT(
            dim = 768,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 30,
            temporal_patch_size = 15,
            spatial_depth = 6,
            temporal_depth = 6,
            dim_head = 32,
            heads = 8
        )

        # CT-CLIP checkpoint: only its text branch (text_transformer.* and
        # to_text_latent.*) is used below to initialise the frozen text encoder.
        ck = torch.load(CT_CLIP_CKPT, map_location = "cpu")

        textmodelpath = BIOMEDVLP_BERT
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.text_model = BertModel.from_pretrained(textmodelpath)

        nck = {}
        nck = {k.replace("text_transformer.", ""):v for k,v in ck.items() if "text_transformer." in k}
        self.text_model.load_state_dict(nck)

        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)


        ###############################

        self.contrastive_proj_image = nn.Linear(config["hidden_size"], self.aligned_length, bias = False)
        self.contrastive_proj_text = nn.Linear(768, self.aligned_length, bias = False)


        latent_ck = {k.replace("to_text_latent.", ""):v for k,v in ck.items() if "to_text_latent." in k}
        self.contrastive_proj_text.load_state_dict(latent_ck)
        # for local
        self.contrastive_proj_image2 = nn.Linear(768, config["hidden_size"], bias = False)
        self.contrastive_proj_text2 = nn.Linear(config["hidden_size"], self.aligned_length, bias = False)
        self.obver_norm_class = nn.Linear(768, 1)

        self.crossAttentionLayer = QFormer(config["hidden_size"], 4)

        ###############################

        self.selector = DefaultSelector_2(768) # DefaultSelector(768)

        # self.modelp = nn.Identity()
        # print(ck)


        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size

        
        #########################################

        resolution_after = config['image_size']

        
        self.expert_image = nn.Embedding(self.num_expert, config["hidden_size"])
        self.expert_image.apply(init_weights)

        self.expert_text = nn.Embedding(self.num_expert, config["hidden_size"])
        self.expert_text.apply(init_weights)

        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        
        self.vision_extract_layer2 = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])

        # self.copy_params()
        # create the queue
        self.register_buffer("text_queue_sim", torch.ones(self.queue_size) * 10000)
        self.register_buffer("text_queue", torch.randn(self.aligned_length, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(self.num_expert, dtype=torch.long)) 

        self.p_enc_3d = PositionalEncodingPermute3D(config["hidden_size"]) 

        # == End

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["mlm"] > 0: # only use this first
            self.mlm_head = prediction_heads.MLMHead(bert_config)
            self.mlm_head.apply(init_weights)

        # == End  : 2. Build Pre-Training Heads ==

        # == Begin: 3. Load Models ==
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict,
                                                     after=resolution_after,
                                                     patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)
        # == End  : 3. Load Models ==

        # waiting

        m3ae_utils.set_metrics(self)


    def infer_image(self, vision_encoder, vision_extract_layer, vision_contrastive_proj, contrastive_proj_image2, expert, img, u=0):

        #print(img.shape)
        uni_modal_image_feats = vision_encoder(img.unsqueeze(1), return_encoded_tokens=True)  # _, c, h, w, d
        # print(uni_modal_image_feats)
        #print(uni_modal_image_feats.shape)
        b, x1,x2,x3, d = uni_modal_image_feats.shape
        uni_modal_image_feats = uni_modal_image_feats.reshape(b, -1, d)
        b, n, d = uni_modal_image_feats.shape
       
        uni_modal_image_feats = contrastive_proj_image2(uni_modal_image_feats)
                
        x_attentions = []
        y_attentions = []

        # == Begin: extract key image feature
        x, y = expert.weight.unsqueeze(0), uni_modal_image_feats
        for layer_idx, (extract_layer) in enumerate(vision_extract_layer):
            x1 = extract_layer(x, y, output_attentions = True)
            y1 = extract_layer(y, x, output_attentions = True)

            x, y = x1[0], y1[0]
            x = x1[0]
            x_attention = x1[1:]
            y_attention = y1[1:]

            x_attentions.extend(x_attention)
            y_attentions.extend(y_attention)
        #print(y)
        attention_map, _ = x_attention[1].max(1) #.softmax(1)

        _, selected_index = torch.sort(attention_map, 2)
        #print(x.shape)
        
        selected_patch = []
        for ii in range(x.shape[0]):
            selected_patch.append(torch.cat([torch.cat([x[ii:ii+1, expert_i:expert_i + 1, :], y[ii:ii+1, selected_index[ii, expert_i, -9:], :]], 1) for expert_i in range(10)], 1)) # use y or orginal uni_modal_image_feats
        selected_patch = torch.cat(selected_patch, 0) 
        # print(x.shape)
        local_image = vision_contrastive_proj(x).reshape(-1, self.aligned_length)

        selected_patch = torch.cat([selected_patch, x], 1)
        if u==0:
            norm_class_res = self.obver_norm_class(x)[0, :8].reshape(1,8)
            return selected_patch, local_image, norm_class_res, uni_modal_image_feats, x_attention[1].mean((1)), x_attentions, y_attentions
        return selected_patch, local_image, x_attention[1].mean((1))


    def infer_text(self, text_embedding, device):
        
        with torch.no_grad():
            local_text = text_embedding

        local_text = local_text.reshape(-1, self.aligned_length)

        return local_text, text_embedding #local_text_ori

    def infer_text_v2(self, text_model, contrastive_proj, input_ids, maps, device):
        
        # print(text_seg_index)
        #print(index.shape)
        # print(input_ids.shape)
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
            # print(local_text_ori.shape)
            local_text = contrastive_proj(local_text_ori).reshape(-1, self.aligned_length)
        return local_text, local_text_ori #local_text_ori

    def cross_layer(self, query, ifea, vision_extract_layer):
        # == Begin: extract key image feature
        x, y = query, ifea
        for layer_idx, (extract_layer) in enumerate(vision_extract_layer):
            x1 = extract_layer(x, y, output_attentions = True)
            y1 = extract_layer(y, x, output_attentions = True)

            x, y = x1[0], y1[0]
            x = x1[0]
            x_attention = x1[1:]
            y_attention = y1[1:]

            # .extend(x_attention)
            # y_attentions.extend(y_attention)
        return x

    def infer_image(self, expert, query, ifea, vision_extract_layer, cmask, lambda_mmr = 1):

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
        cmask = cmask.reshape(b, 10, n)

        print(attention_map.shape, cmask.shape)
        # print(assign_with_prob_threshold_counts(attention_map)[0])
        # print(attention_map.sum(-1))
        # attention_map: [b, 10, 4096]
        if False:
            min_vals = attention_map.min(dim=2, keepdim=True)[0]
            max_vals = attention_map.max(dim=2, keepdim=True)[0]
            attention_map = (attention_map - min_vals) / (max_vals - min_vals + 1e-8)  # 防止除0
            topk_values, topk_indices = torch.topk(attention_map, k=30, dim=2)
            # print(topk_values[0], topk_values.shape, attention_map.shape)
            # print((attention_map[0]>=0.003).sum(1))
            print((attention_map[0]>=0.003).sum(1))

            _, selected_index = torch.sort(attention_map, 2)
            for expert_i in range(10):
                print(attention_map[0, expert_i, selected_index[0, expert_i, -9:]])
                # print(y[0, selected_index[0, expert_i, -9:], :])
                print(feature_similarity_matrix(ifea[0, selected_index[0, expert_i, -9:], :]))
        
        selected_patch, details = select_features_network_mmr_variable_k_gpu_purefeature_r(y, attention_map, query, training = self.training, M = 200, selector_model = self.selector, return_details = True, k_cur = 20)
        
        final_f = []

        if self.training:
            selected_patch = details['soft_agg']
            # print(selected_patch.shape)
            # print([len(selected_patch[0][l]) for l in range(10)], np.sum([len(selected_patch[0][l]) for l in range(10)]))
            
            
            for iii in range(b):
                
                final_f.append(torch.cat([l.unsqueeze(0) for l in selected_patch[iii]], 0).unsqueeze(0))
        else:
            for iii in range(b):
                final_f.append(torch.cat([l.unsqueeze(0) for l in selected_patch[iii]], 0).mean(1).unsqueeze(0))

        return torch.cat(final_f, 0)

    def _per_organ(self, sims, bs = 1):
        new_length = bs * self.num_expert
        sim_text_now = sims[:, :new_length].reshape(bs, self.num_expert, bs, self.num_expert)
        sim_text_queue = sims[:, new_length:].reshape(bs, self.num_expert, self.num_expert, self.queue_size // self.num_expert)
        
        total_text = []
        for l in range(self.num_expert):
            total_text.append(torch.cat([sim_text_now[:, l, :, l], sim_text_queue[:, l, l, :]], 1).unsqueeze(0))
        total_text = torch.cat(total_text, 0)

        return total_text

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
        
        bs = img.shape[0]

        text_ids = batch[f"text_ids"]
        #text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_seg_index = batch[f"seg_index"]

        obver_label = batch[f"obver_label"]
        device = text_ids.device

        cmask = batch[f"mask"]
        print(cmask.shape)

        local_text, local_text_word = self.infer_text(batch["text_embedding"], device)

        local_text_i, local_text_word_i = self.infer_text_v2(self.text_model, self.contrastive_proj_text, batch["all_encodings_i"], batch["all_maps_i"], device)
        local_text_word_i = local_text_word_i.reshape(bs, 10, local_text_word_i.shape[-1])
        impression_idx = batch["impression_seg"]

        D = local_text_word_i # torch.where(impression_idx.squeeze(1).unsqueeze(-1) == 0, local_text_word_i, self.expert_text.weight.unsqueeze(0).repeat(local_text_word_i.shape[0], 1, 1))
        # ret['local_text'] = local_text
        ret['local_text_ori'] = local_text_word

        # with torch.no_grad():
        #     selected_patch, local_image, norm_class_res, uni_modal_image_feats, attention_map, x_all_attn, y_all_attn = self.infer_image(self.vision_encoder, self.vision_extract_layer, self.contrastive_proj_image, self.contrastive_proj_image2, self.expert_image, img.cuda())
        uni_modal_image_feats = batch["all_image"].squeeze(1) #img.cuda().squeeze(1)
        # print(uni_modal_image_feats.shape)
        # print(uni_modal_image_feats.shape) # bs 2048 768
        # print(D.shape)
        if False:
            local_image = self.contrastive_proj_image(self.crossAttentionLayer(D, uni_modal_image_feats, uni_modal_image_feats, add_residual = False)).reshape(-1, self.aligned_length)
        elif False:
            
            local_image = self.contrastive_proj_image(self.cross_layer(D, uni_modal_image_feats, self.vision_extract_layer2)).reshape(-1, self.aligned_length)
        else:
            local_image = self.contrastive_proj_image(self.infer_image(self.expert_image, D, uni_modal_image_feats, self.vision_extract_layer, cmask)).reshape(-1, self.aligned_length)
                
        # ret["attention_map"] = attention_map
        bs = img.shape[0]
        ret['local_image'] = F.normalize(local_image, dim = -1)
        ret['local_text'] = F.normalize(local_text, dim = -1)
        ret['img'] = img
        ret['obver_label'] = obver_label
        ret['selected_patch'] = local_image
        ret['image_feature'] = uni_modal_image_feats
        ret['bs'] = bs
        # ret['x_all_attn'] = x_all_attn
        # ret['y_all_attn'] = y_all_attn 

        with torch.no_grad():
            local_text_m = local_text.detach()
            local_text_m = F.normalize(local_text_m, dim = -1)
            local_text_m_all = torch.cat([local_text_m.t(),self.text_queue.clone().detach()],dim=1)

        ret['local_image_word'] = self.all_gather(uni_modal_image_feats, sync_grads = True)
        ret['local_text_word'] = self.all_gather(local_text_word, sync_grads=True)
        ret['local_text_m_all'] = local_text_m_all
        length = text_seg_index[0][-2]
        ret['length'] = self.all_gather(length)
        

        # ret["norm_class_res"] = norm_class_res[0, :8].reshape(1,8)
        ret["norm_class_label"] = obver_label[0].unsqueeze(0).float()
        # == End  : Text Encoding ==

        # == Begin: extract seg text feature

        

        # == End: extract seg text feature

        # == Begin: Image Encoding ==

        

        # 计算文↔文相似度矩阵
        sim_text = local_text_m @ local_text_m_all
        sim_targets = torch.zeros(sim_text.size()).to(img.device)
        sim_targets.fill_diagonal_(1)       

        total_text = self._per_organ(sim_text, bs)

        soft_target = F.softmax(total_text / self.temp, dim=2)
        # soft_target = F.softmax(sim_text / self.temp, dim=1)  # 归一化后的软目标
        # print(soft_target.shape) # 20 10020

        sim_targets = self._per_organ(sim_targets, bs)

        alpha=0.9

        

        sim_i2t_targets = alpha * soft_target + (1 - alpha) * sim_targets

        ret["sim_i2t_targets"] = sim_i2t_targets
        # print(sim_i2t_m.shape, local_text_m.shape, "xxk") # 10 5010, 10 128

        # print(local_text_m.shape)

        # self._dequeue_and_enqueue2(self.all_gather(local_text_m), self.all_gather(sim_text[:, self.num_expert * uni_modal_image_feats.shape[0]:]))

        ret.update({
            # "patched_images": self.patchify(img), # patch3D
            #"text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "mask_loss": torch.tensor(0.0, device=img.device)
            #"extended_image_masks": extended_image_masks,
            #"extended_text_masks": extended_text_masks,
            #"multi_modal_text_feats": decoder_output,
            #"multi_modal_image_feats": multi_modal_image_feats,
            #"multi_modal_cls_feats": multi_modal_cls_feats,
            #"decoder_output": decoder_output,
            #"text_ori": text_ori,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm5(self, batch))
        
        if "rg" in self.current_tasks:
            ret.update(objectives.compute_rg(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])
        return total_loss


    def on_train_epoch_end(self):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def on_test_epoch_end(self):
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
    def _dequeue_and_enqueue2(self, text_feats, sim):
        # gather keys before updating queue
       
        # print(sim.shape, text_feats.shape) # numver-gpu / bs*10 / 5040, number-gpu / bs * 10 / 128
        text_feats = text_feats.reshape(-1, self.num_expert, self.aligned_length)
        sim = sim.reshape(-1, self.num_expert, self.num_expert, self.queue_size // self.num_expert).mean(-1)
        sim = torch.cat([sim[:, l : l+1, l : l+1] for l in range(self.num_expert)], 1).squeeze()
        # print(text_feats, sim.shape)
        step = self.queue_size // 10
        for idx in range(self.num_expert):
            #print(self.text_queue[:, idx * step : (idx+1) * step].T.shape)
            #print(text_feats.shape)
            cqueue, cscore = self._update_queue(self.text_queue[:, idx * step : (idx+1) * step].T, self.text_queue_sim[idx * step : (idx+1) * step], text_feats[:, idx], sim[:, idx])
            
            #p#rint(cqueue.shape)
            self.text_queue[:, idx * step : (idx+1) * step] = cqueue.T
            self.text_queue_sim[idx * step : (idx+1) * step] = cscore


    def _update_queue(self, text_queue, scores, x, x_scores):
        """
        更新队列，将新特征插入到队列中，并根据分数进行排序和截断。

        参数:
        text_queue (torch.Tensor): 当前的特征队列，维度为 [queue_length, c]。
        scores (torch.Tensor): 当前的分数数组，维度为 [queue_length]。
        x (torch.Tensor): 新的特征，维度为 [b, c]。
        x_scores (torch.Tensor): 新的分数，维度为 [b].

        返回:
        (torch.Tensor, torch.Tensor): 更新后的特征队列和分数数组。
        """
        #print(text_queue.shape, x.shape)
        #print(scores.shape, x_scores.shape)
        # 合并队列和新特征
        combined_features = torch.cat((text_queue, x), dim=0)
        combined_scores = torch.cat((scores, x_scores), dim=0)
        #print(combined_features.shape)
        # 根据分数进行排序
        sorted_indices = torch.argsort(combined_scores, descending=False)
        # print(sorted_indices)
        sorted_features = combined_features[sorted_indices]
        sorted_scores = combined_scores[sorted_indices]
        # print(sorted_features.shape)
        # print(text_queue.size())
        # 截断队列
        updated_text_queue = sorted_features[:text_queue.size(0)]
        updated_scores = sorted_scores[:scores.size(0)]

        return updated_text_queue, updated_scores