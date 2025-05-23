import pytorch_lightning as pl
import torch
import torch.nn as nn


import math
import torch.nn.functional as F

import numpy as np


from CTRG.modules import objectives, m3ae_utils
from CTRG.modules.language_encoders.bert_model import BertCrossLayer
from CTRG.modules.m3ae_utils import init_weights

from CTRG.modules.models.med import BertConfig, BertLMHeadModel

#######################################
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead

# from transformer_maskgit import CTViT
from ctvit import CTViT

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


class M3AETransformerSS_3D_lmae_rg_pretrain_v29(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.momentum = 0.99
        self.queue_size = 10000
        self.temp = 0.07
        self.num_expert = 10
        self.aligned_length = 512
        self.alpha=0.2

      
        self.bs = config['batch_size']
        mask_ori = (torch.ones(self.num_expert, self.num_expert) * 0.1).fill_diagonal_(1).unsqueeze(0)
        mask1 = mask_ori.repeat(self.bs*self.bs, 1, 1).reshape(self.bs, self.bs, self.num_expert, self.num_expert).permute(0,2,1,3).reshape(self.bs*self.num_expert, self.bs*self.num_expert)
        mask2 = mask_ori.repeat(self.bs*self.queue_size // self.num_expert, 1, 1).reshape(self.bs, self.queue_size // self.num_expert, self.num_expert, self.num_expert).permute(0,2,3,1).reshape(self.bs*self.num_expert, self.num_expert*self.queue_size // self.num_expert)
        
        self.imt_mask = torch.cat([mask1, mask2], 1)

        print(self.imt_mask.shape)


        loaded_data = np.load('/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/text_latent_feature.npz')

        # to evaluate the performance of pretrain, we use retrieval metric
        loaded_names = loaded_data['names']
        self.test_text_features = torch.tensor(loaded_data['features'][300:800]) # F.normalize(torch.tensor(loaded_data['features'][300:800]), dim = -1)                
       
        # == Begin: 1. Build Models ==
        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        bert_config = AutoConfig.from_pretrained(textmodelpath)
        # == vision encoder ==


        self.vision_encoder = CTViT(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 30,
            temporal_patch_size = 15,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        )

        
        # load pretrained parameter in CT-CLIP
        ck = torch.load("/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert/CT_CLIP_zeroshot.pt", map_location = "cpu")
        nck = {}
        nck = {k.replace("visual_transformer.", ""):v for k,v in ck.items() if "visual_transformer." in k}

        self.vision_encoder.load_state_dict(nck, strict = False)


        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

        self.text_model = BertModel.from_pretrained(textmodelpath)


        nck = {}
        nck = {k.replace("text_transformer.", ""):v for k,v in ck.items() if "text_transformer." in k}
        self.text_model.load_state_dict(nck)
        ###############################

        self.contrastive_proj_image = nn.Linear(config["hidden_size"], self.aligned_length, bias = False)
        self.contrastive_proj_text = nn.Linear(config["hidden_size"], self.aligned_length, bias = False)

        # for local CL (if adopt)
        self.contrastive_proj_image2 = nn.Linear(512, config["hidden_size"], bias = False)
        self.contrastive_proj_text2 = nn.Linear(config["hidden_size"], self.aligned_length, bias = False)
        self.obver_norm_class = nn.Linear(768, 1)

        ###############################

        
        latent_ck = {k.replace("to_text_latent.", ""):v for k,v in ck.items() if "to_text_latent." in k}
        self.contrastive_proj_text.load_state_dict(latent_ck)

        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size

        # == End  : 1. Build Models ==

        # == Begin: 1.5 

        # class token
        
        self.expert_image = nn.Embedding(self.num_expert, config["hidden_size"])
        self.expert_image.apply(init_weights)

        self.expert_text = nn.Embedding(self.num_expert, config["hidden_size"])
        self.expert_text.apply(init_weights)

        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        

        # self.copy_params()
        # create the queue
        self.register_buffer("text_queue_sim", torch.ones(self.queue_size) * 10000) #
        self.register_buffer("text_queue", torch.randn(self.aligned_length, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(self.num_expert, dtype=torch.long)) 


        m3ae_utils.set_metrics(self)


    def infer_image(self, vision_encoder, vision_extract_layer, vision_contrastive_proj, contrastive_proj_image2, expert, img, u=0):

        uni_modal_image_feats = vision_encoder(img.unsqueeze(1), return_encoded_tokens=True)  # _, c, h, w, d
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
        attention_map, _ = x_attention[1].max(1) #.softmax(1)

        _, selected_index = torch.sort(attention_map, 2)
        
        selected_patch = []
        for ii in range(x.shape[0]):
            selected_patch.append(torch.cat([torch.cat([x[ii:ii+1, expert_i:expert_i + 1, :], y[ii:ii+1, selected_index[ii, expert_i, -9:], :]], 1) for expert_i in range(10)], 1)) # use y or orginal uni_modal_image_feats
        selected_patch = torch.cat(selected_patch, 0) 


        local_image = vision_contrastive_proj(x).reshape(-1, self.aligned_length)

        selected_patch = torch.cat([selected_patch, x], 1)

        if u==0:
            norm_class_res = self.obver_norm_class(x)[0, :8].reshape(1,8)
            return selected_patch, local_image, norm_class_res, uni_modal_image_feats, x_attention[1].mean((1)), x_attentions, y_attentions
        return selected_patch, local_image, x_attention[1].mean((1))

    def infer_text(self, text_model, contrastive_proj, input_ids, maps, device):
        
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        maps = maps.reshape(-1, input_ids.shape[-1])
        with torch.no_grad():
            
            local_text = text_model(input_ids, attention_mask = maps)[0][:, 0:1, :].permute(1,0,2)

            local_text_ori = local_text.reshape(-1, local_text.shape[-1])
        
            local_text = contrastive_proj(local_text).reshape(-1, self.aligned_length)
        return local_text, local_text_ori #local_text_ori

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
        
        text_ids = batch[f"text_ids"]
        text_masks = batch[f"text_masks"]
        text_seg_index = batch[f"seg_index"]

        obver_label = batch[f"obver_label"]
        device = text_ids.device
        # == End  : Fetch the inputs ==

        # == Begin  : Text Encoding ==
        local_text, local_text_word = self.infer_text(self.text_model, self.contrastive_proj_text, batch["all_encodings"], batch["all_maps"], device)

        ret['local_text_ori'] = local_text_word

        selected_patch, local_image, norm_class_res, uni_modal_image_feats, attention_map, x_all_attn, y_all_attn = self.infer_image(self.vision_encoder, self.vision_extract_layer, self.contrastive_proj_image, self.contrastive_proj_image2, self.expert_image, img.cuda())
                
        ret["attention_map"] = attention_map
        ret['local_image'] = F.normalize(local_image, dim = -1)
        ret['local_text'] = F.normalize(local_text, dim = -1)
        ret['img'] = img
        ret['obver_label'] = obver_label
        ret['selected_patch'] = selected_patch
        ret['image_feature'] = uni_modal_image_feats
        ret['x_all_attn'] = x_all_attn
        ret['y_all_attn'] = y_all_attn 

        
            

        with torch.no_grad():
            local_text_m = local_text.detach()
            local_text_m = F.normalize(local_text_m, dim = -1)
            local_text_m_all = torch.cat([local_text_m.t(),self.text_queue.clone().detach()],dim=1)

        ret['local_image_word'] = self.all_gather(uni_modal_image_feats, sync_grads = True)
        ret['local_text_word'] = self.all_gather(local_text_word, sync_grads=True)
        ret['local_text_m_all'] = local_text_m_all
        length = text_seg_index[0][-2]
        ret['length'] = self.all_gather(length)
        

        ret["norm_class_res"] = norm_class_res[0, :8].reshape(1,8)
        ret["norm_class_label"] = obver_label[0].unsqueeze(0).float()
        

        sim_i2t_m = local_text_m @ local_text_m_all / self.temp 

        sim_targets = torch.zeros(sim_i2t_m.size()).to(img.device)
        sim_targets.fill_diagonal_(1)          

        

        

        sim_i2t_targets = self.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - self.alpha) * sim_targets

        ret["sim_i2t_targets"] = sim_i2t_targets


        self._dequeue_and_enqueue2(self.all_gather(local_text_m), self.all_gather(sim_i2t_m[:, self.num_expert * selected_patch.shape[0]:]))

        ret.update({
            "text_ids": text_ids,
            "text_masks": text_masks,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm3(self, batch))

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
            ret.update(objectives.compute_rg(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)
        # print(self.hparams.config)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])
        # print(output, total_loss)
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
    def _dequeue_and_enqueue(self, text_feats, sim):
        # gather keys before updating queue
       
        #print(sim.shape, text_feats.shape) # numver-gpu / bs*10 / 5040, number-gpu / bs * 10 / 128
        text_feats = text_feats.reshape(-1, self.num_expert, self.aligned_length)
        sim = sim.reshape(-1, self.num_expert, self.num_expert, self.queue_size // self.num_expert).sum(-1)
        sim = torch.cat([sim[:, l : l+1, l : l+1] for l in range(self.num_expert)], 1)
        #print(text_feats.shape, sim.shape)
        for idx in range(text_feats.shape[0]):
            # obverlabel = obverlabels[idx]
            text_feat = text_feats[idx]

            for iidx in range(10):
            # obverlabel = torch.cat([obverlabel, torch.zeros(1, device = obverlabel.device)], 0)
                ctext_feat = text_feat[iidx:iidx+1]
                ptr = int(self.queue_ptr[iidx])
                # assert self.queue_size % batch_size == 0  # for simplicity

                step = self.queue_size // 10
                # replace the keys at ptr (dequeue and enqueue)
                if ptr + 1 > step:
                    ptr = 0

                self.text_queue[:, iidx * step + ptr : iidx * step + ptr + 1] = ctext_feat.T.detach()
                ptr = (ptr + 1) % self.queue_size  # move pointer

                self.queue_ptr[iidx] = ptr    

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