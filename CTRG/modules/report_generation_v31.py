import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, BertTokenizer, AutoConfig
import math
import torch.nn.functional as F


from CTRG.modules import objectives, m3ae_utils
from CTRG.modules.language_encoders.bert_model import BertCrossLayer
from CTRG.modules.m3ae_utils import init_weights


from CTRG.modules.models.med import BertConfig, BertModel, BertLMHeadModel
from CTRG.modules.models.vis_utils import save_attention_maps, compute_combined_attention, save_average_cross_attention, compute_last_token_attention_proportion

import numpy as np
#######################################
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead
from transformers import LlamaForCausalLM, LlamaTokenizer


from scipy.special import expit
# from transformer_maskgit import CTViT
from ctvit import CTViT
import sys
sys.path.append("/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/text_classifier")
from classifier import RadBertClassifier

# v14 with 2 layer cross modal path selection
# v15 with feature level retrive enhancement
# v19 with more token

# v20 with high image dimension

# v30 with feature pre extracted

class M3AETransformerSS_3D_lmae_rg_v31(pl.LightningModule):
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
        
        ####for the ce metric cal
        self.tokenizer_ce = AutoTokenizer.from_pretrained('/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/RadBERT', do_lower_case=True)
        self.classificer = RadBertClassifier(n_classes=18)
        self.classificer.load_state_dict(torch.load("/jizhicfs/datalh/dataset/RadBertClassifier.pth"))

        # == vision encoder ==

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

        path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-2-7b-chat-hf"

        self.tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = 0

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        #self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)

        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size

        med_config = '/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/m3ae/modules/configs/med_config_blip.json'
        med_config = BertConfig.from_json_file(med_config)
        # tokenizer 改一下
        self.text_decoder = BertLMHeadModel(config=med_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
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

        self.entropy_record = torch.zeros([1,380])
        self.entropy_sum = torch.ones([1,380])

        # == 4. Build Heads For Downstream Tasks ==
        
        # == End:  4. Build Heads For Downstream Tasks ==


        # == Begin: 5. Load Models For Testing ==
        
        # == End  : 5. Load Models For Testing ==

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

        # == Begin  : Text Encoding ==

        # uni_modal_text_feats = self.text_model(text_ids)['last_hidden_state'] # _, n, c
        # extended_text_masks = self.text_model.get_extended_attention_mask(text_masks, text_masks.size(), device)
        
        # == End  : Text Encoding ==

        # == Begin: extract seg text feature
        
        # == End: extract seg text feature

        # == Begin: Image Encoding ==
        #with torch.no_grad():
        
        selected_patch = img.squeeze(1)
        # print(selected_patch.shape)
        # selected_patch = selected_patch[:, -10:]

        # print(selected_patch.shape)

        #ret["norm_class_res"] = norm_class_res[0, :8].reshape(1,8)
        #ret["norm_class_label"] = obver_label[0].unsqueeze(0).float()
        #ret["attention_map"] = attention_map

        retrive_fea = []
        #for ret_fea in batch["clip_memory"]:
        ## print(batch["clip_memory"])
        #    retrive_fea.append(ret_fea)
        #retrive_fea = torch.cat(retrive_fea, 0)
        #selected_patch = torch.cat([selected_patch, retrive_fea], 1)
        # == End  : Image Encoding ==

        image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                 device=device)
            
       
        # == Begin: Assign Type Embeddings ==
        #uni_modal_text_feats, selected_patch = (
        #    uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
        #    selected_patch + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        #)
        # == End  : Assign Type Embeddings ==

        # == Begin: Multi-Modal Fusion ==
        # ***********************

        text_input = batch["text_ids"].squeeze(1)
        # print(text_input.shape)
        # print(text_input.shape)
        # text_input = torch.cat(text_input, 0)
        # print(batch["rg_encoding"])
        # print(text_labels.input_ids.shape)
        # text_input[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text_input.masked_fill(text_input == self.tokenizer.pad_token_id, -100)

        decoder_targets[:,:0] = -100
        # print(image_masks.shape, attention_map.shape) # 1,45   //  1,96,9
        # selected_patch[:, 5:-1, :] = selected_patch[:, -1:, :]
        decoder_output = self.text_decoder(text_input,
                                           attention_mask = batch["text_masks"].squeeze(1),
                                           encoder_hidden_states = selected_patch,
                                           encoder_attention_mask = image_masks,
                                           labels = decoder_targets,
                                           return_dict = True,
                                           output_attentions = True,
                                          )
        # print(decoder_output.keys())
        print(len(decoder_output["attentions"]))
        if False:
            self_attention_maps = []
            cross_attention_maps = []
            for idx, l in enumerate(decoder_output["attentions"]):
                if idx % 2:
                    cross_attention_maps.append(l.detach().cpu().numpy())
                else:
                    self_attention_maps.append(l.detach().cpu().numpy())

            

            rollout_self_attention, accumulated_cross_attention = compute_combined_attention(self_attention_maps, cross_attention_maps)

            # 保存attention maps
            save_attention_maps(rollout_self_attention, accumulated_cross_attention, save_dir='./attention_maps')

            save_average_cross_attention(accumulated_cross_attention, save_path='./attention_maps/average_cross_attention.png')


            # exit(0)
            #print(decoder_output.shape)
            #print(decoder_output[:, :10, :])
            #exit(0)
            #print(text_masks)
            #print(decoder_output.logits.shape, batch["text_masks"].squeeze(1).shape)
            # 对 logits 进行 softmax 操作，得到概率分布
            probabilities = F.softmax(decoder_output.logits, dim=-1)

            # 计算熵
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)

        #self.entropy_record += entropy.cpu()
        #self.entropy_sum += batch["text_masks"].squeeze(1).cpu()
        # print(entropy)
        #np.save("entropy.npy", self.entropy_record.numpy() / self.entropy_sum.numpy())


        # == Begin: == Output Multi-Modal Features ==
        if not self.training:
            model_kwargs = {"encoder_hidden_states": selected_patch, "encoder_attention_mask":image_masks}
            input_ids = batch[f"prompt_ids"].squeeze(1)
            #input_ids[:,0] = self.tokenizer.bos_token_id
            # input_ids = input_ids[:, :-1]

            # print(input_ids.shape, selected_patch.shape)
            
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                    max_length=380,
                                                    min_length=50,
                                                    num_beams=3,
                                                    eos_token_id=self.tokenizer.sep_token_id,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    return_dict_in_generate=True, 
                                                    output_attentions=True,
                                                    repetition_penalty=1.0,
                                                    **model_kwargs)

            for attention in outputs.attentions:
                self_attention_maps = []
                cross_attention_maps = []
                for idx, l in enumerate(attention):
                    if idx % 2:
                        cross_attention_maps.append(l.detach().cpu().numpy())
                    else:
                        self_attention_maps.append(l.detach().cpu().numpy())
                print(compute_last_token_attention_proportion(self_attention_maps, cross_attention_maps))

            

            print(len(outputs.attentions))
            print([len(attention) for attention in outputs.attentions])
            exit(0)

            captions = []
            for output in outputs.sequences:
                # print(output.shape)
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[:])
        else:
            captions = ["I am fxxking man."] * text_ids.shape[0] # for test 
        text_ori = []

        for l in range(text_ids.shape[0]): 
        
            text_ori.append(self.tokenizer.decode(text_ids.squeeze(1)[l], skip_special_tokens=True))

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
            "text_ori": text_ori,
            "captions": captions,
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