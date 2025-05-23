from .cls_melinda_datamodule import CLSMELINDADataModule
from .irtr_roco_datamodule import IRTRROCODataModule
from .pretraining_medicat_datamodule import MedicatDataModule, MedicatDataModule_3D, MedicatDataModule_3D_RATE, MedicatDataModule_3D_RATE_hr, MedicatDataModule_3D_RATE_hr_fea, MedicatDataModule_3D_RATE_hr_fea_dpo, MedicatDataModule_3D_RATE_hr_fea_all, MedicatDataModule_3D_RATE_hr_fea_CLIP
from .pretraining_roco_datamodule import ROCODataModule
from .vqa_medvqa_2019_datamodule import VQAMEDVQA2019DataModule
from .vqa_slack_datamodule import VQASLACKDataModule
from .vqa_vqa_rad_datamodule import VQAVQARADDataModule

_datamodules = {
    "medicat": MedicatDataModule,
    "roco": ROCODataModule,
    "vqa_vqa_rad": VQAVQARADDataModule,
    "vqa_slack": VQASLACKDataModule,
    "vqa_medvqa_2019": VQAMEDVQA2019DataModule,
    "cls_melinda": CLSMELINDADataModule,
    "irtr_roco": IRTRROCODataModule
}
