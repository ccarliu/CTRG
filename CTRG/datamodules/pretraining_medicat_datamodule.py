from .base_datamodule import BaseDataModule
from ..datasets import MedicatDataset
from ..datasets import CTRGDataset, CTRGDataset_RATE, CTRGDataset_RATE_hr, CTRGDataset_RATE_hr_sim, CTRGDataset_RATE_hr_sim_fea, CTRGDataset_RATE_hr_sim_fea_dpo, CTRGDataset_RATE_hr_sim_fea_all, CTRGDataset_RATE_hr_sim_fea_clip
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizerFast,
    RobertaTokenizerFast
)
from transformers import RobertaConfig, RobertaModel, BertTokenizer, BertModel, AutoTokenizer
from transformers import LlamaTokenizer

class MedicatDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MedicatDataset

    @property
    def dataset_cls_no_false(self):
        return MedicatDataset

    @property
    def dataset_name(self):
        return "medicat"


class MedicatDataModule_3D(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        config_m = AutoConfig.from_pretrained(textmodelpath)
        self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        print(config_m)
        self.vocab_size = self.tokenizer.vocab_size


        collator = (
            DataCollatorForWholeWordMask
            if True
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        
        self.setup_flag = False

    @property
    def dataset_cls(self):
        return CTRGDataset

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(split = "train"
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split = "val"
        )
    
    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split = "test"
        )

    @property
    def dataset_cls_no_false(self):
        return CTRGDataset

    @property
    def dataset_name(self):
        return "medicat"

class MedicatDataModule_3D_RATE_hr(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        #self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

        #path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-2-7b-chat-hf"

        #self.tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
        #self.tokenizer.pad_token_id = 0
        # print(config_m)
        self.vocab_size = self.tokenizer.vocab_size


        collator = (
            DataCollatorForWholeWordMask
            if True
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(tokenizer=self.tokenizer, mlm=False, mlm_probability=0.15)
        
        self.setup_flag = False

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(split = "train"
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split = "val"
        )
    
    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split = "test"
        )

    @property
    def dataset_cls_no_false(self):
        return CTRGDataset_RATE

    @property
    def dataset_name(self):
        return "medicat"

class MedicatDataModule_3D_RATE_hr_fea(MedicatDataModule_3D_RATE_hr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)
        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-2-7b-chat-hf"
        #path = "/apdcephfs_cq10/share_1290796/lh/dataset/Meta-Llama-3-8B-Instruct"

        self.tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
        #self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = 0

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim_fea

class MedicatDataModule_3D_RATE_hr_fea_CLIP(MedicatDataModule_3D_RATE_hr_fea):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)
        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

       

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim_fea_clip


class MedicatDataModule_3D_RATE_hr_fea_all(MedicatDataModule_3D_RATE_hr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)
        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-2-7b-chat-hf"

        self.tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = 0

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim_fea_all


class MedicatDataModule_3D_RATE_hr_fea_dpo(MedicatDataModule_3D_RATE_hr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)
        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-2-7b-chat-hf"

        self.tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = 0

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim_fea_dpo



class MedicatDataModule_3D_RATE(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        config_m = AutoConfig.from_pretrained(textmodelpath)
        self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        print(config_m)
        self.vocab_size = self.tokenizer.vocab_size


        collator = (
            DataCollatorForWholeWordMask
            if True
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        
        self.setup_flag = False

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(split = "train"
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split = "val"
        )
    
    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split = "test"
        )

    @property
    def dataset_cls_no_false(self):
        return CTRGDataset_RATE

    @property
    def dataset_name(self):
        return "medicat"

