import functools

import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler

from .gloria_loss.gloria_loss import global_loss, local_loss

from .dist_utils import all_gather
from scipy.special import expit
from sklearn.metrics import classification_report

import numpy as np





def compute_rg(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    
    loss_lm = infer["decoder_output"].loss
    
    #loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    #print(loss_cls, "xxxxxxxxxxxxx")

    ret = {
        "rg_loss": loss_lm,
        "infer": infer,
    }
    
    phase = "train" if pl_module.training else "val"
    # print(getattr(pl_module, f"{phase}_rg_loss").total)
    # print(infer["captions"])
    # print(infer["text_ori"])

    if not pl_module.training:
        store_path = getattr(pl_module, f"ckpath") + ".txt"
        text_file = open(store_path, "a+")
        text_file.writelines("****" + "\n")
        text_file.writelines(infer["captions"][0] + "\n")
        # print(infer["captions"][0])
    
    loss = getattr(pl_module, f"{phase}_rg_loss")(ret["rg_loss"])
    # print(phase ,1)
    #BLEU_1, _, _, ROUGE_L = getattr(pl_module, f"{phase}_rg_BLEU_1")({1:infer["text_ori"][:1]}, {1:infer["captions"][:1]})
    # print(phase ,2)
    # print(BLEU_1, ROUGE_L, loss)
    
    #####################################
    encodings = pl_module.tokenizer_ce(infer["captions"], return_tensors='pt',max_length=512,padding='max_length',truncation=True)
        
    input_ids = encodings['input_ids'].to(infer["text_ids"].device)
    attention_mask = encodings['attention_mask'].to(infer["text_ids"].device)

    with torch.no_grad():
        pred_logits = pl_module.classificer(input_ids,attention_mask).cpu().numpy()

    # print(pred_logits.shape)
    # pred_logits = pred_logits[1:]
    pred_labels = expit(pred_logits)
    
    pred_labels[pred_labels>=0.5]=1
    pred_labels[pred_labels<0.5]=0

    obver_label = infer["obver_label"].cpu().numpy()

    # print(classification_report(obver_label.flatten(), pred_labels.flatten(), output_dict=True))
    cf1 = classification_report(obver_label.flatten(), pred_labels.flatten(), output_dict=True).get('1', {}).get('f1-score', 0.0)

    # print(cf1)
    #########################################

    pl_module.log(f"rg/{phase}/loss", loss)
    pl_module.log(f"rg/{phase}/accuracy", getattr(pl_module, f"{phase}_rg_BLEU_1")(cf1))
    
    return ret

def compute_rg_blue(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    
    loss_lm = infer["decoder_output"].loss
    
    #loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    #print(loss_cls, "xxxxxxxxxxxxx")

    ret = {
        "rg_loss": loss_lm,
        "infer": infer,
    }
    
    phase = "train" if pl_module.training else "val"
    # print(getattr(pl_module, f"{phase}_rg_loss").total)
    # print(infer["captions"])
    # print(infer["text_ori"])

    if not pl_module.training:
        store_path = getattr(pl_module, f"ckpath") + ".txt"
        text_file = open(store_path, "a+")
        text_file.writelines("****" + "\n")
        text_file.writelines(infer["captions"][0] + "\n")
        # print(infer["captions"][0])
    
    loss = getattr(pl_module, f"{phase}_rg_loss")(ret["rg_loss"])
    # print(phase ,1)
    BLEU_1, _, _, ROUGE_L = getattr(pl_module, f"{phase}_rg_BLEU_2")({1:infer["text_ori"][:1]}, {1:infer["captions"][:1]})
    # print(phase ,2)
    # print(BLEU_1, ROUGE_L, loss)
    
    #####################################
    

    # print(cf1)
    #########################################

    pl_module.log(f"rg/{phase}/loss", loss)
    pl_module.log(f"rg/{phase}/accuracy", getattr(pl_module, f"{phase}_rg_BLEU_1")(BLEU_1))
    
    return ret



def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
   # mlm_logits = pl_module.mlm_head(infer["multi_modal_text_feats"])
    #mlm_labels = infer["text_labels"]

    #mlm_loss = F.cross_entropy(
    #    mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
    #    mlm_labels.view(-1),
    #    ignore_index=-100,
    #)
    # loss_cls = torch.nn.BCEWithLogitsLoss()(infer["norm_class_res"], infer["norm_class_label"])
    
    #sim_i2t = infer["local_image"] @ infer["local_text_m_all"] / 0.07
    # = infer["local_text"] @ infer["local_image_m_all"] / 0.07
                             
    #loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*infer["sim_i2t_targets"],dim=1).mean()
    #loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*infer["sim_t2i_targets"],dim=1).mean() 

    # loss_ita = (loss_i2t+loss_t2i)/2
    n,b,c,l=infer['local_image_word'].shape
    n,b,c2,l=infer['local_text_word'].shape

    # print(infer["local_text"].shape)

    loss_cl = info_nce(infer["local_text"].reshape(n*b, -1), infer["local_image"].reshape(n*b, -1))
    loss_cl += info_nce(infer["local_image"].reshape(n*b, -1), infer["local_text"].reshape(n*b, -1))
   #  print(infer['local_image_word'].shape, infer['local_image_word'].shape)

   #  print(infer['length'])
    loss_local, loss_local2, attn = local_loss(infer['local_image_word'].reshape(n*b, c, l).permute(0,2,1), infer['local_text_word'].reshape(n*b, c2, l).permute(0,2,1), torch.cat(infer['length'], 0))

    # print((loss_local + loss_local2) / 2) #, loss_cls)

    # exit(0)

    ret = {
        "mlm_loss": loss_local + loss_local2 + loss_cl / 2, # + loss_ita, # (loss_local + loss_local2) / 2 * 0.1 + , # +  , # + loss_cls * 0.3,
        #"mlm_logits": mlm_logits,
        #"mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    
    # print(loss_cl)
    # print(infer["local_text"].shape)
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = -1*(loss_local + loss_local2)
    pl_module.log(f"mlm/{phase}/loss", loss_local + loss_local2)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    
    return ret

def compute_mlm3(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    
    sim_i2t = infer["local_image"] @ infer["local_text_m_all"] / 0.07
    imt_mask = pl_module.imt_mask.to(sim_i2t.device)
                             
    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*infer["sim_i2t_targets"]*imt_mask[:sim_i2t.shape[0], :sim_i2t.shape[1]],dim=1).mean()


    loss_ita = loss_i2t



    if torch.isnan(loss_ita):
        loss_ita = infer["local_image"].sum() * 0

    ret = {
        "mlm_loss": loss_ita, # + loss_ita, # (loss_local + loss_local2) / 2 * 0.1 + , # +  , # + loss_cls * 0.3,
        "mlm_ids": infer["text_ids"],
    }

    ### cal the retrieval accuracy
    local_text = infer["local_text"].reshape(-1, 10, pl_module.aligned_length)
    local_image = infer["local_image"].reshape(-1, 10, pl_module.aligned_length)
    

    # print(local_text.shape, local_image.shape, pl_module.test_text_features.shape)
    acc = 0
    for l in range(local_text.shape[0]):
        test_text_feature = torch.cat([local_text[l:l+1].cpu(), pl_module.test_text_features], 0)
        sim = torch.topk(F.cosine_similarity(local_image[l:l+1].cpu(), test_text_feature, dim=-1).sum(-1), 50).indices
        # print(sim)
        if 0 in sim:
            acc += 1
    
    acc = acc / local_text.shape[0]

    phase = "train" if pl_module.training else "val"
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(acc)
    pl_module.log(f"mlm/{phase}/loss", loss_ita)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    
    return ret



def compute_mim(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)

    if pl_module.hparams.config["mim_layer"] == -1:
        multi_modal_image_feats = infer["multi_modal_image_feats"]
    else:
        layer_idx = pl_module.hparams.config["mim_layer"]
        multi_modal_image_feats = infer[f"multi_modal_image_feats_{layer_idx}"]

    mim_logits = pl_module.mim_head(multi_modal_image_feats, infer["mim_ids_restore"])

    target = infer["patched_images"]
    if pl_module.hparams.config["norm_pix_loss"]:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
    mim_labels = target
    mask = infer["mim_masks"]

    mim_loss = (mim_logits - mim_labels) ** 2
    mim_loss = mim_loss.mean(dim=-1)  # [N, L], mean loss per patch
    mim_loss = (mim_loss * mask).sum() / mask.sum()  # mean loss on removed patches

    ret = {
        "mim_loss": mim_loss,
        "mim_logits": mim_logits,
        "mim_labels": mim_labels,
    }
    

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mim_loss")(ret["mim_loss"])
    acc = -loss
    pl_module.log(f"mim/{phase}/loss", loss)
    pl_module.log(f"mim/{phase}/accuracy", acc)

    return ret


def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_head(infer["multi_modal_cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_vqa(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_head(infer["multi_modal_cls_feats"])
    vqa_targets = torch.zeros(len(vqa_logits), pl_module.hparams.config["vqa_label_size"]).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]
    vqa_answer_types = torch.tensor(batch["answer_types"]).to(pl_module.device)

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1])

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
        "vqa_answer_types": vqa_answer_types,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(ret["vqa_logits"], ret["vqa_targets"], ret["vqa_answer_types"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_cls(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    cls_logits = pl_module.cls_head(infer["multi_modal_cls_feats"])
    cls_labels = batch["cls_labels"]
    cls_loss = F.cross_entropy(cls_logits, cls_labels)

    ret = {
        "cls_loss": cls_loss,
        "cls_logits": cls_logits,
        "cls_labels": cls_labels,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_cls_loss")(ret["cls_loss"])
    acc = getattr(pl_module, f"{phase}_cls_accuracy")(ret["cls_logits"], ret["cls_labels"])
    pl_module.log(f"cls/{phase}/loss", loss)
    pl_module.log(f"cls/{phase}/accuracy", acc)

    return ret


def compute_irtr(pl_module, batch, test=False):
    is_training_phase = pl_module.training
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack([batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = torch.stack([batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = torch.stack([batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    batch_infer = {
        "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
        "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
        "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
        "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
    }

    infer = pl_module.infer(batch_infer)

    score = pl_module.irtr_head(infer["multi_modal_cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {"irtr_loss": irtr_loss}

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=256,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(text_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(image_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    # TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # == Begin: Add New Keys ==
        batch_text_preload = {
            "text_ids": _b["text_ids"].to(pl_module.device),
            "text_masks": _b["text_masks"].to(pl_module.device),
            "text_labels": _b["text_labels"].to(pl_module.device),
            "img_index": _b["img_index"],
        }
        text_preload.append(batch_text_preload)
        # == End  : Add New Keys ==

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                # == Begin: Add New Keys ==
                batch_infer = {
                    "text_ids": txt_batch["text_ids"],
                    "text_masks": txt_batch["text_masks"],
                    "text_labels": txt_batch["text_labels"],
                }
                score = pl_module.irtr_head(pl_module.infer(batch_infer, img=im, )["multi_modal_cls_feats"])[:, 0]
                # == End  : Add New Keys ==

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)
