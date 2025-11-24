from .base_datamodule import BaseDataModule
from ..datasets import  CTRGDataset_RATE_hr_sim, CTRGDataset_RATE_hr_sim_fea
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizerFast,
    RobertaTokenizerFast
)
from transformers import RobertaConfig, RobertaModel, BertTokenizer, BertModel, AutoTokenizer
from transformers import LlamaTokenizer




class MedicatDataModule_3D_RATE_hr(BaseDataModule):
    def __init__(self, _config):
        super().__init__(_config)

        self._config = _config
        textmodelpath = _config["text_tokenlizer_path"]
        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

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
        self.train_dataset = self.dataset_cls(split = "train", config = self._config
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split = "val", config = self._config
        )
    
    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split = "test", config = self._config
        )

    @property
    def dataset_cls_no_false(self):
        return CTRGDataset_RATE

    @property
    def dataset_name(self):
        return "medicat"

class MedicatDataModule_3D_RATE_hr_fea(BaseDataModule):
    def __init__(self, _config):
        super().__init__(_config)

        self._config = _config
        path = _config["text_tokenlizer_path"]
        self.tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = 0

        self.setup_flag = False

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim_fea
    
    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(split = "train", config = self._config
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split = "val", config = self._config
        )
    
    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split = "test", config = self._config
        )