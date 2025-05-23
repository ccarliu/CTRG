import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertConfig, BertModel

from m3ae.modules import objectives, m3ae_utils
from m3ae.modules import prediction_heads
from m3ae.modules.language_encoders.bert_model import BertCrossLayer
from m3ae.modules.m3ae_utils import init_weights
from m3ae.modules.vision_encoders import swin_transformer as swin
from m3ae.modules.vision_encoders.clip_model import build_model, adapt_position_encoding
from m3ae.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding

from m3ae.modules.models.med import BertConfig, BertModel, BertLMHeadModel

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep

#######################################
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead


class M3AETransformerSS_3D_lmae_rg(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

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
            '''
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
            '''
            textmodelpath = "/apdcephfs/share_1290796/lh/dataset/CTRG/model"
            bert_config = AutoConfig.from_pretrained(textmodelpath)
        else:
            raise ValueError

        # == Begin: 1. Build Models ==
        

        # == vision encoder ==
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.vision_encoder = SwinViT(
            in_chans=1,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
        )

        pretrainpath = "/apdcephfs/share_1290796/lh/dataset/CTRG/model/model_swinvit.pt"
        state_dict = torch.load(pretrainpath)['state_dict']
        state_dict = {k.replace("module.", "").replace("fc", "linear"):v for k,v in state_dict.items()}
        # self.vision_encoder.load_state_dict(state_dict, strict = False)

        textmodelpath = "/apdcephfs/share_1290796/lh/dataset/CTRG/model"
        config_m = AutoConfig.from_pretrained(textmodelpath)
        self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)

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

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_language_layers.apply(init_weights)

        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        # == End  : 1. Build Models ==

        # == Begin: 1.5 
        
        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        
        self.text_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        
        '''
        self.vision_another_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        '''
        '''
        # class token
        parameter = nn.Parameter(torch.randn(1, 9, config['hidden_size']))
        setattr(self, f'expert', parameter)
        '''
        self.expert_image = nn.Embedding(9, config["hidden_size"])
        self.expert_image.apply(init_weights)

        self.expert_text = nn.Embedding(9, config["hidden_size"])
        self.expert_text.apply(init_weights)
        

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

    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        pos_embed = self.vision_encoder.visual.positional_embedding.unsqueeze(0).to(x)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x += pos_embed[:, 1:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        x_ = x_ + pos_embed[:, :1]
        x_masked = torch.cat((x_, x_masked), dim=1)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        p = self.hparams.config["patch_size"]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.hparams.config["patch_size"]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

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
            img = batch[img_key][0]
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_seg_index = batch[f"seg_index"]
        text_ori = batch[f"text"]
        device = text_ids.device
        # == End  : Fetch the inputs ==

        # == Begin: Text Encoding ==
        # print("!!!!!!!!!!!!!!!", text_ids.shape)
        #print(batch[img_key].shape)
        uni_modal_text_feats = self.text_model(text_ids)['last_hidden_state'] # _, n, c
        extended_text_masks = self.text_model.get_extended_attention_mask(text_masks, text_masks.size(), device)
        
        # == End  : Text Encoding ==

        # == Begin: Image Encoding ==
        
        uni_modal_image_feats = self.vision_encoder(img.contiguous())[4]  # _, c, h, w, d
        b, c, h, w, d = uni_modal_image_feats.shape
        # print(uni_modal_image_feats.shape)
        uni_modal_image_feats = uni_modal_image_feats.permute(0, 2,3,4, 1).reshape(b, h*w*d, c)
        
        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)

        
        # print(uni_modal_image_feats.shape, uni_modal_text_feats.shape) # 
        # == End  : Image Encoding ==

        # == Begin: extract key image feature

        # x, y = self.expert_image.weight.unsqueeze(0), uni_modal_image_feats
        '''
        x, y = self.expert, uni_modal_image_feats
        for layer_idx, (extract_layer) in enumerate(self.vision_extract_layer):
            x1 = extract_layer(x, y, output_attentions = True)
            y1 = extract_layer(y, x, output_attentions = True)

            x, y = x1[0], y1[0]
            x_attention = x1[1:]
            y_attention = y1[1:]


        attention_map = x_attention[1].mean(1)
        _, selected_index = torch.sort(attention_map, 2)

        ret["selected_index"] = selected_index
        ret["attention_map"] = attention_map

        selected_patch = torch.cat([y[:, selected_index[0, expert_i, -5:], :] for expert_i in range(9)], 1)

        '''
        # == Begin: extract key image feature
        x, y = self.expert_text.weight.unsqueeze(0), uni_modal_image_feats
        for layer_idx, (extract_layer) in enumerate(self.vision_extract_layer):
            x1 = extract_layer(x, y, output_attentions = True)
            y1 = extract_layer(y, x, output_attentions = True)

            x, y = x1[0], y1[0]
            x_attention = x1[1:]
            y_attention = y1[1:]


        attention_map = x_attention[1].mean(1)

        # print(attention_map.shape) 1,9,
        

        _, selected_index = torch.sort(attention_map, 2)


        #print(y_attention[1].shape)
        # print(y_attention[1].shape)

        ret["selected_index"] = selected_index
        ret["attention_map"] = attention_map

        selected_patch = torch.cat([uni_modal_image_feats[:, selected_index[0, expert_i, -5:], :] for expert_i in range(9)], 1)

        # == End: extract key image feature

        image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                 device=device)
            
        extended_image_masks = self.text_model.get_extended_attention_mask(image_masks, image_masks.size(), device)
        extended_image_masks_cs = extended_image_masks.repeat(1,1,extended_text_masks.shape[-1], 1)

        for idx_idx, idx in enumerate(text_seg_index[0][1:]):
            # c_seg = text_seg_index[0][idx_idx] : idx
            extended_image_masks_cs[0,0,text_seg_index[0][idx_idx] : idx, :idx_idx*5] = -10000
            extended_image_masks_cs[0,0,text_seg_index[0][idx_idx] : idx, (idx_idx+1)*5:] = -10000

        # print(image_masks, extended_image_masks, extended_text_masks)
        # print(text_seg_index, len(text_seg_index), len(text_seg_index[0]))

        #print(x.shape, y.shape) # 1 9 768, 1 96 768
        #print(x_attention[0].shape, x_attention[1].shape) # 1, 12, 9, 9, //  1, 12, 9, 96
        #print(y_attention[0].shape, y_attention[1].shape)
        #print(selected_patch.shape)
        #exit(0)
            

        # == Begin: Assign Type Embeddings ==
        uni_modal_text_feats, selected_patch = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            selected_patch + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        # == End  : Assign Type Embeddings ==

        # == Begin: Multi-Modal Fusion ==

            # == End  : For visualization: Return the attention weights ==
        # == End  : Multi-Modal Fusion ==

        # == Begin: == Output Multi-Modal Features ==
        # == End  : == Output Multi-Modal Features ==


        # == Begin: == Output report generator 
        # print(text_ids.shape)
        text_labels = batch["rg_encoding"][0]
        # print(batch["rg_encoding"])
        # print(text_labels.input_ids.shape)
        text_labels[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text_labels.masked_fill(text_labels == self.tokenizer.pad_token_id, -100)

        decoder_targets[:,:0] = -100

        decoder_output = self.text_decoder(text_labels,
                                           attention_mask = batch["rg_attn"][0],
                                           encoder_hidden_states = selected_patch,
                                           encoder_attention_mask = image_masks,
                                           labels = decoder_targets,
                                           return_dict = True,
                                          )
        
        # generate text:
        
        model_kwargs = {"encoder_hidden_states": selected_patch, "encoder_attention_mask":image_masks}
        input_ids = batch[f"prompt_ids"][0]
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=380,
                                                  min_length=50,
                                                  num_beams=3,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=1.0,
                                                  **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[1:])
        '''
        captions = ["I am fxxking man."] # for test 
        '''
        # == End: == Output report generator 

        ret.update({
            "images": img,
            # "patched_images": self.patchify(img), # patch3D
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "extended_image_masks": extended_image_masks,
            "extended_text_masks": extended_text_masks,
            # "multi_modal_text_feats": multi_modal_text_feats,
            # "multi_modal_image_feats": multi_modal_image_feats,
            # "multi_modal_cls_feats": multi_modal_cls_feats,
            "decoder_output": decoder_output,
            "text_ori": text_ori,
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
            ret.update(objectives.compute_mlm(self, batch))

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

    def training_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return m3ae_utils.set_schedule(self)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
