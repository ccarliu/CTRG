import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, BertTokenizer, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import math
import torch.nn.functional as F


from CTRG.modules import objectives, m3ae_utils
from CTRG.modules.language_encoders.bert_model import BertCrossLayer
from CTRG.modules.m3ae_utils import init_weights
from transformers import LlamaForCausalLM, LlamaTokenizer

from CTRG.modules.models.med import BertConfig, BertModel, BertLMHeadModel

from peft import get_peft_model, LoraConfig, TaskType

import matplotlib.pyplot as plt


#######################################
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead


# from transformer_maskgit import CTViT
from ctvit import CTViT
from classifier import RadBertClassifier

# v14 with 2 layer cross modal path selection
# v15 with feature level retrive enhancement
# v19 with more token

# v20 with high image dimension

# v30 with llm

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

def analyze_image_attention(attention_maps, image_token_start, image_token_end):
    """
    分析每个生成 token 对输入图像 token 的 attention 比重。

    参数:
        attention_maps (list): 每一层的 attention map 列表，每个 attention map 的形状为 [1, 32, 1, k]。
        image_token_start (int): 图像 token 的起始位置。
        image_token_end (int): 图像 token 的结束位置。

    返回:
        dict: 每个生成 token 对图像 token 的 attention 比重，格式为 {token_index: attention_weights}。
    """
    results = {}

    for token_index, layer_attention_maps in enumerate(attention_maps[1:]):
        # 初始化存储每个 token 对图像 token 的 attention 比重
        image_attention_weights = []

        for layer_idx, attention_map in enumerate(layer_attention_maps[:]):
            # attention_map 的形状为 [1, 32, 1, k]
            # 取最后一个维度（k）中对应图像 token 的部分
            image_attention = attention_map[:, :, :, image_token_start:image_token_end].sum(-1) / attention_map[:, :, :, :].sum(-1)

            # 计算每个注意力头对图像 token 的 attention 均值
            # head_attention_weights, _ = image_attention.max(dim=-1)  # 形状为 [1, 32, 1]
            
            head_attention_weights = image_attention
            image_attention_weights.append(head_attention_weights)

        # 将所有层的 attention 权重拼接并计算均值
        image_attention_weights = torch.cat(image_attention_weights, dim=1)  # 形状为 [1, 32 * num_layers, 1]
        image_attention_weights = image_attention_weights.mean(dim=1).squeeze()  # 形状为 [1]

        # 存储结果
        results[token_index] = image_attention_weights.mean().cpu()

    return results

class M3AETransformerSS_3D_lmae_rg_v32(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.momentum = 0.99
        self.queue_size = 512
        self.temp = 0.07
        self.aligned_length = 128
        self.selected_patch_n = config["selected_patch"]
        print(self.selected_patch_n)
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for those features (should include diagnosis of following abnormities: Medical material,Arterial wall calcification,Cardiomegaly,Pericardial effusion,Coronary artery wall calcification,Hiatal hernia,Lymphadenopathy,Emphysema,Atelectasis,Lung nodule,Lung opacity,Pulmonary fibrotic sequela,Pleural effusion,Mosaic attenuation pattern,Peribronchial thickening,Consolidation,Bronchiectasis,Interlobular septal thickening) selected from a chest CT image.'
        #
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for those features selected from a chest CT image.'
        
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

            textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
            bert_config = AutoConfig.from_pretrained(textmodelpath)
        else:
            raise ValueError

        # == Begin: 1. Build Models ==
        
        # == vision encoder ==

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        #self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)

        

        path = config["decoder_path"]
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/Meta-Llama-3-8B-Instruct"
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
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.05,target_modules=["W_pack", "o_proj"]
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


        self.multi_modal_vision_proj_forllm = SimpleMLP(config['hidden_size'], 4096) # 2560 for 4b, 2048 for 1.8b, 1024for.5b, 1536 for 2 1.5b 896
        self.multi_modal_vision_proj_forllm.apply(init_weights)
        
        # == End  : 1. Build Models ==

        # == Begin: 1.5 
        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        
        self.text_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(1)])
        
        self.contrastive_proj_image = nn.Linear(config["hidden_size"], 128)
        self.contrastive_proj_image2 = nn.Linear(512, config["hidden_size"])
        self.contrastive_proj_text = nn.Linear(config["hidden_size"], 128)
        self.obver_norm_class = nn.Linear(768, 1)

        # class token
        
        self.expert_image = nn.Embedding(10, config["hidden_size"])
        self.expert_image.apply(init_weights)

        self.expert_text = nn.Embedding(10, config["hidden_size"])
        self.expert_text.apply(init_weights)
        
        self.image_position_embeddings = nn.Embedding(96, config["hidden_size"])


        # == End

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["mlm"] > 0: # only use this first
            self.mlm_head = prediction_heads.MLMHead(bert_config)
            self.mlm_head.apply(init_weights)
        '''
        if config["loss_names"]["mim"] > 0:
            self.mim_head = prediction_heads.MIMHead(config)
            self.mim_head.apply(init_weights)
        if config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"]["irtr"] > 0:
            self.itm_head = prediction_heads.ITMHead(config["hidden_size"] * 2)
            self.itm_head.apply(init_weights)
        '''
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

        # == 4. Build Heads For Downstream Tasks ==
        
        # == End:  4. Build Heads For Downstream Tasks ==


        # == Begin: 5. Load Models For Testing ==
        
        # == End  : 5. Load Models For Testing ==

    def prompt_wrap(self, img_embeds, atts_img, cls_res = None):

        if cls_res is not None:
            abnorm = [score.cpu().numpy() for score in cls_res[0].detach().sigmoid()]
            struct_list = ["Thoracic strctures", "Rib", "Lung", "Heart", "Pleural", "Liver", "Kidney", "Thyroid"]
            abnorm_str = ""
            for iii, struct in enumerate(struct_list):
                abnorm_str += struct + ": " + str(abnorm[iii]) + "; "
            cprompt  = self.prompt + " The abnorm probability of each of eight structure are: " + abnorm_str
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

        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)

        # print(p_before_embeds.shape, img_embeds.shape, p_after_embeds.shape)

        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

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
        # selected_patch = x

        # selected_patch = uni_modal_image_feats * attention_map.unsqueeze(-1)
        if u==0:
            norm_class_res = self.obver_norm_class(x)[0, :8].reshape(1,8)
            return selected_patch, local_image, norm_class_res, x_attention[1].mean((1))
        return selected_patch, local_image, x_attention[1].mean((1))



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
        # == End  : Fetch the inputs ==

        selected_patch = img.squeeze(1)
        b, n, l = selected_patch.shape
        selected_patch = self.multi_modal_vision_proj_forllm(selected_patch)



        image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                 device=device)
        
        img_embeds, atts_img = self.prompt_wrap(selected_patch, image_masks)
        
        text_input = batch["text_ori"]
        to_regress_tokens = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=380,
            add_special_tokens=False
        ).to(device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)


        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)

        decoder_output = self.text_decoder(inputs_embeds=inputs_embeds.to(device),
                                           attention_mask=attention_mask.to(device),
                                           return_dict=True,
                                           labels=targets.to(device),
                                          )

        # == Begin: == Output Multi-Modal Features ==
        if not self.training:
            outputs = self.text_decoder.generate(
                inputs_embeds=img_embeds,
                num_beams=1,
                do_sample=False,
                min_new_tokens=50,
                max_new_tokens=330,
                return_dict_in_generate=True, 
                output_attentions=True,
            )

            captions = []
            for output in outputs.sequences:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[:])
        else:
            captions = ["I am fxxking man."] # for test 



        ret.update({
            "images": img,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "multi_modal_text_feats": decoder_output,
            "decoder_output": decoder_output,
            "text_ori": text_input,
            "captions": captions,
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
            ret.update(objectives.compute_rg_blue(self, batch))

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