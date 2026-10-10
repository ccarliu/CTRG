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
from .multimodal_encoder.vit import ViT

# from transformer_maskgit import CTViT
from ctvit import CTViT
from .m3ae_utils import feature_similarity_matrix

from paths_config import (
    BIOMEDVLP_BERT,
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

def mask_features_torch(uni_modal_image_feats, mask_rate=0.5):
    b, n, l = uni_modal_image_feats.size()
    n_half = int(n * (1-mask_rate))

    # 为每个样本生成一个随机的索引序列
    indices = torch.randperm(n)

    # 选择前 n_half 个索引并排序以保持顺序
    selected_indices = indices[:n_half].sort().values

    # 根据选择的索引提取特征
    masked_feats = uni_modal_image_feats[:, selected_indices, :]

    return masked_feats

class M3AETransformerSS_3D_lmae_rg_pretrain_v29_Qwen_perorgan_mask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.momentum = 0.99
        self.queue_size = 10000
        self.temp = 0.07
        self.num_expert = 10
        # self.embedding_length = 1024
        self.aligned_length = 512
        self.mask_rate = config['mask_rate']
      
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

        if 'bert' in config['tokenizer'] or True:
            textmodelpath = CTRG_MODEL_PATH
            bert_config = AutoConfig.from_pretrained(textmodelpath)
        else:
            raise ValueError
        # == Begin: 1. Build Models ==
        textmodelpath = CTRG_MODEL_PATH
        config_m = AutoConfig.from_pretrained(textmodelpath)

        # == vision encoder ==
        self.vision_enc_type = 1

        if self.vision_enc_type == 1:

            self.vision_encoder = CTViT(
                dim = config["hidden_size"],
                codebook_size = 8192,
                image_size = 480,
                patch_size = 30,
                temporal_patch_size = 15,
                spatial_depth = 6,
                temporal_depth = 6,
                dim_head = 32,
                heads = 8
            )
        elif self.vision_enc_type == 2:
            self.vision_encoder = ViT(
                image_size = (480, 480),          # image size
                frames = 240,               # max number of frames
                image_patch_size = 32,     # image patch size
                frame_patch_size = 16,      # frame patch size
                dim = 768,
                depth = 12,
                heads = 8,
                mlp_dim = 2048,
                dropout = 0.1,
                channels = 1,
                emb_dropout = 0.1
            )
        else:
            self.vision_encoder = ViT(
                in_channels=1,
                img_size=[32,256,256],
                patch_size=[8,16,16],
                hidden_size=768,
                mlp_dim=3072,#3072,
                num_layers=12,
                num_heads=12,
                pos_embed="perceptron",
                dropout_rate=0,
                spatial_dims=3,
                classification=False,
            )

        textmodelpath = BIOMEDVLP_BERT
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)


        ###############################

        self.contrastive_proj_image = nn.Linear(768, self.aligned_length, bias = False)
        # self.contrastive_proj_text = nn.Linear(4096, self.aligned_length, bias = False)

        # for local
        self.contrastive_proj_image2 = nn.Linear(config["hidden_size"], 768, bias = False)
        self.contrastive_proj_text2 = nn.Linear(768, self.aligned_length, bias = False)
        self.obver_norm_class = nn.Linear(768, 1)

        ###############################

        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size

        
        #########################################

        resolution_after = config['image_size']

        
        self.expert_image = nn.Embedding(self.num_expert, 768)
        self.expert_image.apply(init_weights)

        self.expert_text = nn.Embedding(self.num_expert, 768)
        self.expert_text.apply(init_weights)

        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        

        # self.copy_params()
        # create the queue
        self.register_buffer("text_queue_sim", torch.ones(self.queue_size) * 10000)
        self.register_buffer("text_queue", torch.randn(self.aligned_length, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(self.num_expert, dtype=torch.long)) 

        # self.p_enc_3d = PositionalEncodingPermute3D(config["hidden_size"]) 

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

    def sparse_negative_entropy_loss(self, attention_map, cmask, tau=0.0003, eps=1e-8):
        """
        attention_map: (B, T, S) 已经归一化到 [0,1]
        cmask:         (B, T, S) 0 表示需要监督的位置
        tau:           阈值，只惩罚 > tau 的位置
        """
        # 1. 只关注 mask==0 的区域
        inverse_mask = (cmask == 0)

        # 2. 再筛选出 attention > tau 的位置
        hard_mask = (attention_map > tau) & inverse_mask

        # 3. 如果没有满足条件的位置，直接返回 0
        num_hard = hard_mask.sum()
        if num_hard == 0:
            return torch.tensor(0.0, device=attention_map.device, requires_grad=True)

        # 4. 只在这些位置上计算 -log(1-p)
        attention_map = torch.clamp(attention_map, min=eps, max=1 - eps)
        loss = -torch.log(1 - attention_map) / 0.1 * hard_mask.float()

        # 5. 平均
        return loss.sum() / num_hard

    def infer_image(self, vision_encoder, vision_extract_layer, vision_contrastive_proj, contrastive_proj_image2, expert, img, cmask, u=0):

        #print(img.shape)
        if self.vision_enc_type == 1:
            uni_modal_image_feats = vision_encoder(img.unsqueeze(1), return_encoded_tokens=True)  # _, c, h, w, d # 240 480 480
            b, x1,x2,x3, d = uni_modal_image_feats.shape
            uni_modal_image_feats = uni_modal_image_feats.reshape(b, -1, d)
        else:
            # print(img.shape)
            all_images = []
            for l in range(6):
                # print(img.shape)
                uni_modal_image_feats, _ = vision_encoder(img.unsqueeze(1)[:, :, l*32:(l+1)*32])  # _, c, h, w, d
                # print(uni_modal_image_feats.shape)
                all_images.append(uni_modal_image_feats)
            uni_modal_image_feats = torch.cat(all_images, 1)

            bad = torch.isnan(uni_modal_image_feats) | torch.isinf(uni_modal_image_feats)
            if bad.sum() > 0:
                # 填充 0 并 detach，梯度畅通
                uni_modal_image_feats = torch.where(bad, torch.zeros_like(uni_modal_image_feats).detach(), uni_modal_image_feats)
                # 打印提示（只打印一次 / 每 step，可自行加 rank 过滤）
                print(f"[WARN] contains NaN/Inf - {bad.sum().item()} elements ")
        # print(uni_modal_image_feats.shape)
        b, n, d = uni_modal_image_feats.shape

        if self.training and False:
            uni_modal_image_feats = mask_features_torch(contrastive_proj_image2(uni_modal_image_feats), self.mask_rate)
        else:
            uni_modal_image_feats = contrastive_proj_image2(uni_modal_image_feats)
                
        x_attentions = []
        y_attentions = []

        # == Begin: extract key image feature
        x, y = expert.weight.unsqueeze(0), uni_modal_image_feats
        for layer_idx, (extract_layer) in enumerate(vision_extract_layer):
            # extract_layer.eval()
            x1 = extract_layer(x, y, output_attentions = True)
            y1 = extract_layer(y, x, output_attentions = True)

            x, y = x1[0], y1[0]
            x = x1[0]
            x_attention = x1[1:]
            y_attention = y1[1:]
            
            x_attentions.extend(x_attention[1])
            y_attentions.extend(y_attention[1])
        # p rint(y)
        # attention_map, _ = x_attention[1].max(1) #.softmax(1)
        attention_map = x_attention[1].mean(1) #.softmax(1)
        cmask = cmask.reshape(b, 10, n)
        # print(cmask.shape, attention_map.shape)
        # print(attention_map.max(), attention_map.min())
        ################
        # 只监督 mask == 0 的位置
        loss =  self.sparse_negative_entropy_loss(attention_map[:, :9], cmask[:, :9])
        ################
        

        _, selected_index = torch.sort(attention_map, 2)
        # print(attention_map.sum(2))
        # print((attention_map[0]>=0.003).sum(1))
        # for expert_i in range(10):
          #   print(attention_map[0, expert_i, selected_index[0, expert_i, -9:]])

        #print(x.shape)
        
        selected_patch = []
        for ii in range(x.shape[0]):
            # for expert_i in range(10):
             #   print(feature_similarity_matrix(y[ii, selected_index[ii, expert_i, -9:], :]))
            selected_patch.append(torch.cat([torch.cat([x[ii:ii+1, expert_i:expert_i + 1, :], y[ii:ii+1, selected_index[ii, expert_i, -9:], :]], 1) for expert_i in range(10)], 1)) # use y or orginal uni_modal_image_feats
        selected_patch = torch.cat(selected_patch, 0) 
        # print(x.shape)
        local_image = vision_contrastive_proj(x).reshape(-1, self.aligned_length)
        bad = torch.isnan(local_image) | torch.isinf(local_image)
        if bad.sum() > 0:
            # 填充 0 并 detach，梯度畅通
            local_image = torch.where(bad, torch.zeros_like(local_image).detach(), local_image)
            print("prelocalimage bad", bad.sum())

        selected_patch = torch.cat([selected_patch, x], 1)
        if u==0:
            norm_class_res = self.obver_norm_class(x)[0, :8].reshape(1,8)
            return selected_patch, local_image, norm_class_res, uni_modal_image_feats, x_attention[1].mean(1), x_attentions, y_attentions, loss, cmask
        return selected_patch, local_image, x_attention[1].mean((1))


    def infer_text(self, text_embedding, device):
        
        with torch.no_grad():
            local_text = text_embedding

        local_text = local_text.reshape(-1, self.aligned_length)

        return local_text, text_embedding #local_text_ori


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
        if self.vision_enc_type == 3:
            img = batch["image_downsample"]

        text_ids = batch[f"text_ids"]
        #text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_seg_index = batch[f"seg_index"]

        obver_label = batch[f"obver_label"]

        cmask = batch[f"mask"]

        device = text_ids.device

        local_text, local_text_word = self.infer_text(batch["text_embedding"], device)

        ret['local_text_ori'] = local_text_word

        selected_patch, local_image, norm_class_res, uni_modal_image_feats, attention_map, x_all_attn, y_all_attn, mask_loss, cmask = self.infer_image(self.vision_encoder, self.vision_extract_layer, self.contrastive_proj_image, self.contrastive_proj_image2, self.expert_image, img.cuda(), cmask)
        
        #print(local_image.shape)
        ret["attention_map"] = attention_map
        ret['local_image'] = F.normalize(local_image, dim = -1) # , eps=1e-8)
        ret['local_text'] = F.normalize(local_text, dim = -1) # , eps=1e-8)
        ret['img'] = img
        ret['mask'] = cmask
        ret['obver_label'] = obver_label
        ret['selected_patch'] = selected_patch
        ret['image_feature'] = uni_modal_image_feats
        ret['x_all_attn'] = x_all_attn
        ret['y_all_attn'] = y_all_attn 
        ret['bs'] = bs
        ret['mask_loss'] = mask_loss
        return ret

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
        

        # 计算文↔文相似度矩阵
        sim_text = local_text_m @ local_text_m_all
        sim_targets = torch.zeros(sim_text.size()).to(img.device)
        sim_targets.fill_diagonal_(1)       

        total_text = self._per_organ(sim_text, bs)

        soft_target = F.softmax(total_text / self.temp, dim=2)
        # soft_target = F.softmax(sim_text / self.temp, dim=1)  # 归一化后的软目标
        # print(soft_target.shape) # 20 10020

        sim_targets = self._per_organ(sim_targets, bs)

        alpha=1

        

        sim_i2t_targets = alpha * soft_target + (1 - alpha) * sim_targets

        ret["sim_i2t_targets"] = sim_i2t_targets

        self._dequeue_and_enqueue2(self.all_gather(local_text_m), self.all_gather(sim_text[:, self.num_expert * selected_patch.shape[0]:]))

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