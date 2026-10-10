import io
import os
import random

import pyarrow as pa
import torch
from PIL import Image

import SimpleITK as sitk
import re
import numpy as np
from ..transforms import keys_to_transforms

import json  

import csv
from scipy.ndimage import rotate, zoom

from transformers import RobertaConfig, RobertaModel, BertTokenizer, AutoConfig, AutoTokenizer
import pickle

import torch.nn.functional as F

import nltk
import glob

import nibabel as nib
# nltk.download('punkt')  # 下载punkt模型，用于句子分割

from nltk.tokenize import sent_tokenize

from paths_config import (
    CTRG_ANNOTATION_JSON, BIOMEDVLP_BERT,
    CTRATE_TRAIN_REPORTS, CTRATE_VALID_REPORTS,
    CTRATE_TRAIN_LABELS, CTRATE_VALID_LABELS,
    CTRATE_TRAIN_REGION_REPORT, CTRATE_VALID_REGION_REPORT,
    CTRATE_TRAIN_IMAGES, CTRATE_VALID_IMAGES,
    CTRATE_TRAIN_MASK_FAST, CTRATE_VALID_MASK_FAST,
    TEXT_EMBEDDING_DIR, FEATURE_PATH_TRAIN, FEATURE_PATH_VALID,
    JSON_PATH_TRAIN, JSON_PATH_VALID,
)
path = CTRG_ANNOTATION_JSON
# 打开文件并读取内容  
with open(path, 'r', encoding='utf-8') as file:  
    # 使用json.load()方法解析JSON数据  
    data = json.load(file) 

import unicodedata
import string
def is_punctuation(ch: str) -> bool:
    """
    判断字符 ch 是否为标点符号。
    方法：使用 unicodedata.category 判断 Unicode 类别是否以 'P' 开头（P = punctuation）。
    适用于常见的 ASCII 标点和许多 Unicode 标点（如中文标点）。
    """
    if not ch or len(ch) != 1:
        raise ValueError("请输入单个字符")
    # 快速判断 ASCII 常见标点（可选加速）
    if ch in string.punctuation:
        return True
    # Unicode 分类以 'P' 开头表示标点（如 'Pc','Pd','Pe','Pf','Pi','Po' 等）
    return unicodedata.category(ch).startswith('P')

class CTRGDataset_RATE_hr_sim_fea(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380
        self.target_size = [240, 480, 480]

        self.pretrain = False

        # Image Transformations

        if self.usage == "train":
            self.text_path = CTRATE_TRAIN_REPORTS
            self.image_path = CTRATE_TRAIN_IMAGES
            self.text_embedding_path = TEXT_EMBEDDING_DIR
            self.label_path = CTRATE_TRAIN_LABELS
            self.json_path = JSON_PATH_TRAIN
            self.feature_path = FEATURE_PATH_TRAIN
            self.region_report_path = CTRATE_TRAIN_REGION_REPORT
            self.mask_fast_path = CTRATE_TRAIN_MASK_FAST
        else:
            self.text_path = CTRATE_VALID_REPORTS
            self.image_path = CTRATE_VALID_IMAGES
            self.text_embedding_path = TEXT_EMBEDDING_DIR
            self.label_path = CTRATE_VALID_LABELS
            self.json_path = JSON_PATH_VALID
            self.feature_path = FEATURE_PATH_VALID
            self.region_report_path = CTRATE_VALID_REGION_REPORT
            self.mask_fast_path = CTRATE_VALID_MASK_FAST

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        self.text = {}
        self.label = {}
        all_text = {}
        all_label = {}
        self.all_clinic_info = {}
        self.all_impression = {}

        self.text_embedding = []

        textmodelpath = BIOMEDVLP_BERT
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = BertModel.from_pretrained(textmodelpath)
        self.tokenizer_text_model = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)
        

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_text[cline[0]] = cline[3]
            self.all_clinic_info[cline[0]] = cline[1]
            self.all_impression[cline[0]] = cline[4]
            

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            start_id = 0
            bs = 20
            
        elif self.usage == "train":
            start_id = 0
            bs = 40000000000
        else: 
            start_id = 0
            bs = 3000

        print(len(list(all_text.keys())))
        
        idx = 0
        for l in list(all_text.keys()):
            
            key = l
            if len(all_text[l]) < 500:
                continue
            # image_path = os.path.join(self.feature_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz" + "")
            image_path = os.path.join(self.feature_path, key.split(".")[0] + ".npz" + "selected_patch.npy") #"selected_patch.npy")
            #print(image_path)
            
            if os.path.exists(image_path):
                idx += 1
                if idx < start_id:
                    continue
                if idx >= start_id + bs:
                    break

                self.text[key] = all_text[key]
                if self.pretrain:
                    self.text_embedding.append(np.load(os.path.join(self.text_embedding_path, key.split(".")[0] + "_text_embedding.npy")))
                self.label[key] = all_label[key]

        self.text_embedding = np.array(self.text_embedding)

        print(len(self.text.keys()))

         # get region report
        column1 = []
        column2 = []
        column3 = []
        column4 = []
        with open(self.region_report_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # 读取第一行（列名）
            
            # 读取每一行数据
            for row in csv_reader:
                column1.append(row[0])
                column2.append(row[1])
                column3.append(row[2])
                #@ if self.usage != "train":
                column4.append(row[3])

        all_name = list(set(column1))
        all_region = list(set([l for l in column2 if l != "" and "/" not in l]))
        
        self.name_2_region = {}
        self.name_2_abnorm = {}

        for patient in all_name:
            self.name_2_region[patient] = {}
            self.name_2_abnorm[patient] = {}

        for iidx, patient in enumerate(column1):
            if column2[iidx] == "" or "/" in column2[iidx]:
                continue
            self.name_2_region[patient][column2[iidx]] = column3[iidx]
            # if self.usage != "train":
            self.name_2_abnorm[patient][column2[iidx]] = column4[iidx]

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.text.keys())

    def get_raw_image(self, index, image_key="image"):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path
    
    def get_image_fea(self, key):

        image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "selected_patch.npy")
        all_feature = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "image_feature.npy")

        selected_patch = np.load(image_path)
        return selected_patch, image_path, np.load(all_feature)

    def pad_or_crop(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。
        
        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。
        mode (str): 模式，'train' 表示训练模式，'test' 表示测试模式。
        
        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        if self.usage == 'train':
            # 随机裁剪或填充
            start_depth = np.random.randint(0, max(target_depth - input_depth, 1))
            start_height = np.random.randint(0, max(target_height - input_height, 1))
            start_width = np.random.randint(0, max(target_width - input_width, 1))
        else:
            # 中心裁剪或填充
            start_depth = max((target_depth - input_depth) // 2, 0)
            start_height = max((target_height - input_height) // 2, 0)
            start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]

        return output_tensor

    def augment_image(self, image_array):
        """
        对3D图像进行数据增强。
        
        参数:
        image_array (numpy.ndarray): 3D图像数据
        
        返回:
        augmented_image (numpy.ndarray): 增强后的图像
        """
        # 随机旋转
        angle = np.random.uniform(-10, 10)  # 随机选择旋转角度
        rotated_image = rotate(image_array, angle, axes=(1, 2), reshape=False)
        
        ## 随机缩放
        #zoom_factor = np.random.uniform(0.9, 1.1)  # 随机选择缩放因子
        #zoomed_image = zoom(rotated_image, (1, zoom_factor, zoom_factor))
        
        return rotated_image

    
    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor
    

    def image_prepro(self, image):

        # maybe crop like this
        # image = self.pad_or_crop(self.augment_image(image), target_size = self.target_size)
        image = self.pad_or_crop2(image, target_size = self.target_size)
        image = image.transpose(0,2,1)
        image = torch.tensor(image).float() 

        return image

    def get_mask(self, index, img_shape = None):
        key = list(self.text.keys())[index]
        mask_name = "_".join(key.split(".")[0].split("_")[:-1]) + "_1"
        fast_path = os.path.join(self.mask_fast_path, mask_name + ".npy")
        if not os.path.exists(fast_path):
            raise FileNotFoundError(
                f"Fast structural mask not found: {fast_path}. "
                "Fast masks ship with the data bundle under "
                "$CTRG_DATA_ROOT/ctrate/mask_fast/{train,valid}; check "
                "CTRG_DATA_ROOT / CTRATE_{TRAIN,VALID}_MASK_FAST."
            )
        return torch.tensor(np.load(fast_path))

    def get_image(self, index, image_key="image"):

        if False:
            image, path = self.get_raw_image(index, image_key=image_key)
            image_tensor = self.image_prepro(image)
        else:

            hasp = False
            key = list(self.text.keys())[index]
            image_tensor, path, all_image = self.get_image_fea(key)

            #print(image_tensor.shape)
            image_tensor = torch.tensor(image_tensor)
            # all_image = torch.tensor(all_image)
            if self.pretrain:
                mask = self.get_mask(index)

                image, path = self.get_raw_image(index, image_key=image_key)
                mask = self.get_mask(index, image.shape)

                image = image*1000
                hu_min, hu_max = -1000, 400
                image = np.clip(image, hu_min, hu_max)

                image = (((image +300 ) / 700)).astype(np.float32)

                image_tensor_raw = self.image_prepro(image)
                #################################

        ret = {
            "image": image_tensor,
            # "image_feature_p": image_feature_p,
            # "text_feature_p": text_feature_p,
            "img_index": list(self.text.keys())[index],
            "cap_index": 0,
            # "hasp": hasp, 
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.label[list(self.label.keys())[index]],
            "all_image": all_image,
            # "mask": mask,
            # "image_tensor_raw": image_tensor_raw
        }

        if self.pretrain:
            ret["mask"] = mask
            ret["image_tensor_raw"] = image_tensor_raw

        return ret

    def seg_text(self, text):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        ###### pre seg
        segs = sent_tokenize(text)
        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == []:

                #print(xxx)
                # print(segs[iidx], segs[iidx-1].strip()[-1], iidx)
                #print(segs[iidx].strip()[:11])
                if segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]


                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]

                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # pass
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx])
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += " " + ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += " " + ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += " " + ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += " " + ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += " " + ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += " " + ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += " " + ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += " " + ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += " " + ("No abnormalities in other organs.")
            
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        #print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,

            )
        # print(final_seg)
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))

        #print(final_idx, final_token_merge)
        #for iidx in range(len(final_idx)-4):
        #    print(final_text[final_token_merge["offset_mapping"][final_idx[iidx]][0]:final_token_merge["offset_mapping"][final_idx[iidx+1]][0]])
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def extract_abnorm(self, jpath):
        with open(jpath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        organ_keys = ["trachea and bronchie", "mediastinum", "heart", "lung", "esophagus", "pleura", "bone", "thyroid", "breast", "abdomen", "others"]
        return_sentence = []
        return_sentence_ab = []
        for organ in organ_keys:
            csentence = ""
            if len(data["hierarchical_abnormalities"][organ]["attributes"]["abnormality"]) != 0:
                if len(csentence) == 0:
                    csentence = organ + ", "
                csentence += ", ".join(data["hierarchical_abnormalities"][organ]["attributes"]["abnormality"]) + "."
            ab_sentence = csentence
            if len(data["hierarchical_abnormalities"][organ]["attributes"]["non_abnormality"]) != 0 and False:
                if len(csentence) == 0:
                    csentence = organ + ", "
                csentence += ", ".join(data["hierarchical_abnormalities"][organ]["attributes"]["non_abnormality"]) + "."
            if csentence != "":
                return_sentence.append(csentence)
            if ab_sentence != "":
                return_sentence_ab.append(ab_sentence)

            # print("all", csentence, "ab", ab_sentence)
        
        return_sentence.extend(data["global_abnormalities"])
        return_sentence.extend(data["impression_disorders"])

        return_sentence_ab.extend(data["global_abnormalities"])
        return_sentence_ab.extend(data["impression_disorders"])

        # final_abnorm_sentence = ". ".join(return_sentence) + "."
        final_abnorm_sentence = " ".join(return_sentence)
        final_abnorm_sentence_ab = " ".join(return_sentence_ab)
        # print(final_abnorm_sentence)
        return final_abnorm_sentence, final_abnorm_sentence_ab

    def extract_abnorm_noise(self, jpath):
        with open(jpath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        organ_keys = ["trachea and bronchie", "mediastinum", "heart", "lung", "esophagus", "pleura", "bone", "thyroid", "breast", "abdomen", "others"]
        return_sentence = []
        return_sentence_ab = []
        for organ in organ_keys:
            csentence = ""
            if len(data["hierarchical_abnormalities"][organ]["attributes"]["abnormality"]) != 0:
                if len(csentence) == 0:
                    csentence = organ + ", "
                input_list = data["hierarchical_abnormalities"][organ]["attributes"]["abnormality"]
                input_list = [item for item in input_list if random.random() >= 0.1]
                csentence += ", ".join(input_list) + "."
            ab_sentence = csentence
            if len(data["hierarchical_abnormalities"][organ]["attributes"]["non_abnormality"]) != 0:
                if len(csentence) == 0:
                    csentence = organ + ", "
                input_list = data["hierarchical_abnormalities"][organ]["attributes"]["non_abnormality"]
                input_list = [item for item in input_list if random.random() >= 0.9]
                csentence += ", ".join(input_list) + "."
            if csentence != "":
                return_sentence.append(csentence)
            if ab_sentence != "":
                return_sentence_ab.append(ab_sentence)

            # print("all", csentence, "ab", ab_sentence)
        
        return_sentence.extend(data["global_abnormalities"])
        return_sentence.extend(data["impression_disorders"])

        return_sentence_ab.extend(data["global_abnormalities"])
        return_sentence_ab.extend(data["impression_disorders"])

        # final_abnorm_sentence = ". ".join(return_sentence) + "."
        final_abnorm_sentence = " ".join(return_sentence)
        final_abnorm_sentence_ab = " ".join(return_sentence_ab)
        # print(final_abnorm_sentence)
        return final_abnorm_sentence, final_abnorm_sentence_ab

    def get_seg_text(self, patient_name):
        
        if True:
            cregion_report = self.name_2_region[patient_name]
            cregion_report = self.name_2_abnorm[patient_name]
            # print("xxk", cregion_report)
            if 'mediastinum' in cregion_report.keys():
                if "heart" in cregion_report.keys() and cregion_report["mediastinum"] != "None":
                    cregion_report["heart"] += cregion_report["mediastinum"]
                elif cregion_report["mediastinum"] != "None":
                    cregion_report["heart"] = cregion_report["mediastinum"]
        
            regions = ['trachea and bronchie', 'heart', 'lung', 'esophagus', 'pleura',  'bone', 'thyroid', 'breast', 'abdomen', 'others']

            cregion_report_new = {}

            merge_sentence = ""
            for r in regions:
                if r not in cregion_report.keys() or cregion_report[r] == "None":
                    cregion_report_new[r] = r + ": No abnormality found." 
                elif bool(re.fullmatch(r'[\[\]\s]*', cregion_report[r])):
                    cregion_report_new[r] = r + ": No abnormality found." 
                elif len(cregion_report[r]) <= 4:
                    cregion_report_new[r] = r + ": No abnormality found." 
                else:
                    if is_punctuation(cregion_report[r][-1]):
                        cregion_report[r] = cregion_report[r][:-1] + "."
                    else:
                        cregion_report[r] += "."

                    cregion_report_new[r] = r + ": " + cregion_report[r] #  No abnormality found." 
                    merge_sentence += r + ": " + cregion_report[r] + " "
            
            final_seg = [cregion_report_new[r] for r in regions if r in cregion_report_new]
            return final_seg, merge_sentence
            # return ". ".join(final_seg)
        else:
            json_path_c = os.path.join(self.json_path, patient_name.split(".")[0] + ".json")
            with open(json_path_c, 'r', encoding='utf-8') as f:
                json_c = json.load(f)
            cregion_report = self.extract_all_abnormalities_grouped(json_c)

            final_seg = [cregion_report[r] for r in list(cregion_report.keys())]
        

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        #print(final_text)
        # print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg


    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        key = list(self.text.keys())[raw_index]
        text_ori = self.text[key]

        current_ab_text, current_ab_text_ab = self.extract_abnorm(os.path.join(self.json_path, key.split(".")[0] + ".json"))
        # current_ab_text, _ = self.extract_abnorm_noise(os.path.join(self.json_path, key.split(".")[0] + ".json"))
        
        # extract_ last report

        items = key.split("_")
        if items[2] != "a":
            items[2] = chr(ord(items[2]) - 1)
            newkey = "_".join(items)
            _, last_text = self.extract_abnorm(os.path.join(self.json_path, newkey.split(".")[0] + ".json"))
        else:
            last_text = "Not given."
        
        clinical = self.all_clinic_info[key]
        impression = self.all_impression[key]

        if np.random.rand() > 0.5 and False: # self.usage == "train":
            final_seg_i, merge_sentence = self.get_seg_text(key)
            # print(key, ": ", final_seg_i, merge_sentence)
            impression = merge_sentence
        else:
            impression = current_ab_text_ab

            encoding_i, text_i, final_idx_i, final_seg_i = self.seg_text(current_ab_text)
            # print(key, " --seg: ", final_seg_i)
        
        if len(text_ori.split(" ") ) > 240:
            text = " ".join(text_ori.split(" ")[:220]) + "."
        else:
            text = text_ori
        encoding, text, final_idx, final_seg = self.seg_text(text)
        # encoding, text, final_idx, final_seg = self.get_seg_text(key)
        
        
        
        # print(final_seg_i)
        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

     
        rg_encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt")


        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg': final_seg,
                # "text_embedding": self.text_embedding[raw_index],


                "impression_seg": final_seg_i, # final_seg_i,
                'impression': impression, # impression,
                "clinical": clinical,
                "last_text": last_text,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg': final_seg,
                # "text_embedding": self.text_embedding[raw_index],


                "impression_seg": final_seg_i, # final_seg_i,
                'impression': impression, # impression,
                "clinical": clinical,
                "last_text": last_text,
            }

    def get_label_similarity(self, vector1, vector2):
        intersection = torch.sum((vector1 == 1) & (vector2 == 1)).item()
        union = torch.sum((vector1 == 1) | (vector2 == 1)).item()
        
        if union == 0:
            return 1
        jaccard_similarity = intersection / union
        return jaccard_similarity

    def get_suite(self, index):
        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)

            result = True

        name = list(self.text.keys())[ret["img_index"]]
        name_list = name.split(".")[0].split("_")
        
        return ret
        
    
    def __getitem__(self, index):

        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        new_ret["text_masks"] = ret["rg_attn"]
        new_ret["seg_index"] = torch.tensor(ret["seg_index"])
        new_ret["obver_label"] = ret["obver_label"].long()
        new_ret["prompt_ids"] = ret["prompt_ids"]

        # new_ret["image_feature_p"] = ret["image_feature_p"]
        # new_ret["text_feature_p"] = ret["text_feature_p"]
        new_ret["all_image"] = ret["all_image"]

        # new_ret["hasp"] = torch.tensor(ret["hasp"])

        final_seg = ret["final_seg"]

        all_encoding = []
        for seg in final_seg:
            all_encoding.append(self.tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings = torch.cat([l.input_ids for l in all_encoding])
        all_maps = torch.cat([l.attention_mask for l in all_encoding])
        #print(all_encodings.shape)
        new_ret["all_encodings"] = all_encodings
        new_ret["all_maps"] = all_maps
        new_ret["image"] = ret["image"]
        if self.pretrain:
            new_ret["mask"] = ret["mask"]
            new_ret["image_tensor_raw"] = ret["image_tensor_raw"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text_ori"] = ret["text"][0]

        new_ret["prior"] = "Clinical info: " + ret['clinical'] + " last report abnormality: " + ret['last_text']

        new_ret['impression'] = ret["impression"]

        # tokenlizer impression
        all_encoding = []
        final_seg_i = ret["impression_seg"]
        for seg in final_seg_i:
            all_encoding.append(self.tokenizer_text_model(seg, padding="max_length",
                truncation=True,
                max_length=150,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings_i = torch.cat([l.input_ids for l in all_encoding])
        all_maps_i = torch.cat([l.attention_mask for l in all_encoding])
        new_ret["all_encodings_i"] = all_encodings_i
        new_ret["all_maps_i"] = all_maps_i
        new_ret["impression_seg"] = final_seg_i # torch.tensor([l[:13] == "No abnormalit" for l in final_seg_i]).unsqueeze(0)
        new_ret["impression_seg_text"] = final_seg_i


        # tokenlizer impression rej
        '''
        all_encoding = []
        final_seg_i = ret["impression_seg_rej"]
        for seg in final_seg_i:
            all_encoding.append(self.tokenizer_text_model(seg, padding="max_length",
                truncation=True,
                max_length=150,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings_i = torch.cat([l.input_ids for l in all_encoding])
        all_maps_i = torch.cat([l.attention_mask for l in all_encoding])
        new_ret["all_encodings_i_rej"] = all_encodings_i
        new_ret["all_maps_i_rej"] = all_maps_i
        new_ret["impression_seg_rej"] = final_seg_i # torch.tensor([l[:13] == "No abnormalit" for l in final_seg_i]).unsqueeze(0)
        '''
        if self.pretrain:
            new_ret["text_embedding"] = torch.tensor(self.text_embedding[ret["raw_index"]])
        # new_ret["prior"] = "Clinical info: " + "Not Given." + " last report abnormality: " + "Not Given."
        
        return new_ret
    

class CTRGDataset_RATE_hr_sim_fea_clip(CTRGDataset_RATE_hr_sim_fea):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380
        self.target_size = [240, 480, 480]

        # Image Transformations
        # not implement yet
        #/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1823_e6
        #/jizhicfs/datalh/M3AE/img_feature/version_1887_e6
        #/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1859_e8

        if self.usage == "train":
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/train_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_train_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_train"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_train"
            self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/CLIP"
        else:
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_valid"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_valid"
            self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/CLIP"

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        self.text = {}
        self.label = {}
        all_text = {}
        all_label = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)
        

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_text[cline[0]] = cline[3]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = 100
        else:
            limit = 10000000

        for l in list(all_text.keys()):
            key = l
            if len(all_text[l]) < 500:
                continue
            # image_path = os.path.join(self.feature_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz" + "")
            image_path = os.path.join(self.feature_path, "_".join(key.split(".")[0].split("_")[:2]) + key.split(".")[0].split("_")[2] + "img_ori_embedding.npy")

            #print(image_path)

            #print(image_path)
            if os.path.exists(image_path) and limit > 0:
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]

        print(len(self.text.keys()))

    
    def get_image_fea(self, index):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.feature_path,  "_".join(key.split(".")[0].split("_")[:2]) + key.split(".")[0].split("_")[2] + "img_ori_embedding.npy")

        return np.load(image_path), image_path


class CTRGDataset_RATE_hr_sim_fea_all(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380
        self.target_size = [240, 480, 480]

        # Image Transformations
        # not implement yet
        # /apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1823_e6

        if self.usage == "train":
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/train_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_train_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_train"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_train"
            self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/version_1859_e8"
        else:
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_valid"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_valid"
            self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/version_1859_e8"

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        self.text = {}
        self.label = {}
        all_text = {}
        all_label = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)
        

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_text[cline[0]] = cline[3]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = 300
        else:
            limit = 10000000

        for l in list(all_text.keys()):
            key = l
            if len(all_text[l]) < 500:
                continue
            # image_path = os.path.join(self.feature_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz" + "")
            image_path = os.path.join(self.feature_path, key.split(".")[0] + ".npz" + "selected_patch.npy")

            #print(image_path)
            if os.path.exists(image_path) and limit > 0:
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]

        print(len(self.text.keys()))

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.text.keys())

    def get_raw_image(self, index, image_key="image"):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path
    
    def get_image_fea(self, index):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "image_feature.npy")
        image_path_ = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "selected_patch.npy")

        return np.load(image_path), image_path, np.load(image_path_)

    def pad_or_crop(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。
        
        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。
        mode (str): 模式，'train' 表示训练模式，'test' 表示测试模式。
        
        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        if self.usage == 'train':
            # 随机裁剪或填充
            start_depth = np.random.randint(0, max(target_depth - input_depth, 1))
            start_height = np.random.randint(0, max(target_height - input_height, 1))
            start_width = np.random.randint(0, max(target_width - input_width, 1))
        else:
            # 中心裁剪或填充
            start_depth = max((target_depth - input_depth) // 2, 0)
            start_height = max((target_height - input_height) // 2, 0)
            start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]

        return output_tensor

    def augment_image(self, image_array):
        """
        对3D图像进行数据增强。
        
        参数:
        image_array (numpy.ndarray): 3D图像数据
        
        返回:
        augmented_image (numpy.ndarray): 增强后的图像
        """
        # 随机旋转
        angle = np.random.uniform(-10, 10)  # 随机选择旋转角度
        rotated_image = rotate(image_array, angle, axes=(1, 2), reshape=False)
        
        ## 随机缩放
        #zoom_factor = np.random.uniform(0.9, 1.1)  # 随机选择缩放因子
        #zoomed_image = zoom(rotated_image, (1, zoom_factor, zoom_factor))
        
        return rotated_image

    
    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor
    

    def image_prepro(self, image):

        # maybe crop like this
        # image = self.pad_or_crop(self.augment_image(image), target_size = self.target_size)
        image = self.pad_or_crop2(image, target_size = self.target_size)
        image = image.transpose(0,2,1)
        image = torch.tensor(image).float() 

        return image

    def get_image(self, index, image_key="image"):
        
        if False:
            image, path = self.get_raw_image(index, image_key=image_key)
            image_tensor = self.image_prepro(image)
        else:
            image_tensor, path, select_patch = self.get_image_fea(index)
            #print(image_tensor.shape)
            image_tensor = torch.tensor(image_tensor)

        return {
            "image": image_tensor,
            "select_patch": torch.tensor(select_patch),
            "img_index": list(self.text.keys())[index],
            "cap_index": 0,
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.label[list(self.label.keys())[index]]
        }

    def seg_text(self, text):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        ###### pre seg
        segs = sent_tokenize(text)

        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == []:

                #print(xxx)
                # print(segs[iidx], segs[iidx-1].strip()[-1], iidx)
                #print(segs[iidx].strip()[:11])
                if segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]


                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]

                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += " " + ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += " " + ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += " " + ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += " " + ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += " " + ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += " " + ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += " " + ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += " " + ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += " " + ("No abnormalities in other organs.")
            
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        #print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,

            )
        # print(final_seg)
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))

        #print(final_idx, final_token_merge)
        #for iidx in range(len(final_idx)-4):
        #    print(final_text[final_token_merge["offset_mapping"][final_idx[iidx]][0]:final_token_merge["offset_mapping"][final_idx[iidx+1]][0]])
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        text_ori = self.text[list(self.text.keys())[raw_index]]

       
        if len(text_ori.split(" ") ) > 240:
            text = " ".join(text_ori.split(" ")[:220]) + "."
        else:
            text = text_ori
        encoding, text, final_idx, final_seg = self.seg_text(text)

 
        ''' for finetune
        encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                # return_tensors="pt",
                return_offsets_mapping=True)
        '''

        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

        if True:
            rg_encoding = self.tokenizer(text_ori, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt")
        else:  
            rg_encoding = self.tokenizer(text, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt")
        # print(rg_encoding)
        '''
        self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        '''
        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }

    def get_suite(self, index):
        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
            for i in range(self.draw_false_image): # 0
                ret.update(self.get_false_image(i, selected_index=index))
            for i in range(self.draw_false_text): # 0
                ret.update(self.get_false_text(i, selected_index=index))
            result = True
            #except Exception as e:
            #    print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #    index = random.randint(0, len(self.index_mapper) - 1)

        name = list(self.text.keys())[ret["img_index"]]
        name_list = name.split(".")[0].split("_")
        clip_memory = np.load(os.path.join(self.clip_feature_path, name_list[0] + "_" + name_list[1]+name_list[2] + ".npy"))
        ret["clip_memory"] = torch.tensor(clip_memory).unsqueeze(0)

        llama_feature = np.load(os.path.join(self.llama_feature_path, name_list[0] + "_" + name_list[1]+ "_" +name_list[2] +"_1"+ ".npz_text_feature.npy"))
        ret["llama_feature"] = llama_feature
        
        return ret
    
    def __getitem__(self, index):

        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        new_ret["text_masks"] = ret["rg_attn"]
        new_ret["seg_index"] = torch.tensor(ret["seg_index"])
        new_ret["obver_label"] = ret["obver_label"].long()
        new_ret["prompt_ids"] = ret["prompt_ids"]

        final_seg = ret["final_seg"]

        all_encoding = []
        for seg in final_seg:
            all_encoding.append(self.tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings = torch.cat([l.input_ids for l in all_encoding])
        all_maps = torch.cat([l.attention_mask for l in all_encoding])
        #print(all_encodings.shape)
        new_ret["all_encodings"] = all_encodings
        new_ret["all_maps"] = all_maps
        new_ret["image"] = ret["image"]
        new_ret["select_patch"] = ret["select_patch"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text_ori"] = ret["text"][0]
        
        
        return new_ret
  

class CTRGDataset_RATE_hr_sim_fea_dpo(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380
        self.target_size = [240, 480, 480]

        # Image Transformations
        # not implement yet
        #/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1823_e6

        if self.usage == "train":
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/train_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_train_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_train"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_train"
            self.feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1859_e8"
        else:
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_valid"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_valid"
            self.feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1859_e8"

        #textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        #config_m = AutoConfig.from_pretrained(textmodelpath)
        #self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)

        self.retrieve_name = {}

        self.text = {}
        self.label = {}
        all_text = {}
        all_label = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)
        

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_text[cline[0]] = cline[3]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = 300
        else:
            limit = 10000000

        for l in list(all_text.keys()):
            key = l
            if len(all_text[l]) < 500:
                continue
            # image_path = os.path.join(self.feature_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz" + "")
            image_path = os.path.join(self.feature_path, key.split(".")[0] + ".npz" + "selected_patch.npy")

            #print(image_path)
            if os.path.exists(image_path) and limit > 0:
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]

        print(len(self.text.keys()))

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.text.keys())

    def get_raw_image(self, index, image_key="image"):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path
    
    def get_image_fea(self, index):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "selected_patch.npy")

        # image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "image_feature.npy")

        return np.load(image_path), image_path

    def get_image_fea_2(self, name):
        
        # for retrieve

        # key = list(self.text.keys())[index]

        image_path = os.path.join(self.feature_path,  name + ".npz" + "selected_patch.npy")

        # image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "image_feature.npy")

        return np.load(image_path), image_path

    def pad_or_crop(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。
        
        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。
        mode (str): 模式，'train' 表示训练模式，'test' 表示测试模式。
        
        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        if self.usage == 'train':
            # 随机裁剪或填充
            start_depth = np.random.randint(0, max(target_depth - input_depth, 1))
            start_height = np.random.randint(0, max(target_height - input_height, 1))
            start_width = np.random.randint(0, max(target_width - input_width, 1))
        else:
            # 中心裁剪或填充
            start_depth = max((target_depth - input_depth) // 2, 0)
            start_height = max((target_height - input_height) // 2, 0)
            start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]

        return output_tensor

    def augment_image(self, image_array):
        """
        对3D图像进行数据增强。
        
        参数:
        image_array (numpy.ndarray): 3D图像数据
        
        返回:
        augmented_image (numpy.ndarray): 增强后的图像
        """
        # 随机旋转
        angle = np.random.uniform(-10, 10)  # 随机选择旋转角度
        rotated_image = rotate(image_array, angle, axes=(1, 2), reshape=False)
        
        ## 随机缩放
        #zoom_factor = np.random.uniform(0.9, 1.1)  # 随机选择缩放因子
        #zoomed_image = zoom(rotated_image, (1, zoom_factor, zoom_factor))
        
        return rotated_image

    
    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor
    

    def image_prepro(self, image):

        # maybe crop like this
        # image = self.pad_or_crop(self.augment_image(image), target_size = self.target_size)
        image = self.pad_or_crop2(image, target_size = self.target_size)
        image = image.transpose(0,2,1)
        image = torch.tensor(image).float() 

        return image

    def get_image(self, index, image_key="image"):
        
        if False:
            image, path = self.get_raw_image(index, image_key=image_key)
            image_tensor = self.image_prepro(image)
        else:
            image_tensor, path = self.get_image_fea(index)
            #print(image_tensor.shape)
            image_tensor = torch.tensor(image_tensor)

            key = list(self.text.keys())[index]
            # print(list(self.text.keys())[:10])
            #print(self.retrieve_name[key.split(".")[0] + ".npzlocal"] )
            #print(image_path)   
            #print(list(self.retrieve_name.keys())[index])

            image_tensor_cd, path_cd = self.get_image_fea_2(self.retrieve_name[key.split(".")[0] + ".npzlocal"][10].split(".")[0])
            # image_tensor_cd, path = self.get_image_fea((index + 5) % len(self.text.keys()))
            #print(image_tensor.shape)
            image_tensor_cd = torch.tensor(image_tensor_cd)

        return {
            "image": image_tensor,
            "image_cd": image_tensor_cd,
            "img_index": list(self.text.keys())[index],
            "cap_index": 0,
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.label[list(self.label.keys())[index]]
        }

    def seg_text(self, text):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        ###### pre seg
        segs = sent_tokenize(text)

        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == []:

                #print(xxx)
                # print(segs[iidx], segs[iidx-1].strip()[-1], iidx)
                #print(segs[iidx].strip()[:11])
                if segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]


                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]

                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += " " + ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += " " + ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += " " + ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += " " + ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += " " + ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += " " + ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += " " + ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += " " + ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += " " + ("No abnormalities in other organs.")
            
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        #print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,

            )
        # print(final_seg)
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))

        #print(final_idx, final_token_merge)
        #for iidx in range(len(final_idx)-4):
        #    print(final_text[final_token_merge["offset_mapping"][final_idx[iidx]][0]:final_token_merge["offset_mapping"][final_idx[iidx+1]][0]])
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        text_ori = self.text[list(self.text.keys())[raw_index]]

       
        if len(text_ori.split(" ") ) > 240:
            text = " ".join(text_ori.split(" ")[:220]) + "."
        else:
            text = text_ori
        encoding, text, final_idx, final_seg = self.seg_text(text)

 
        ''' for finetune
        encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                # return_tensors="pt",
                return_offsets_mapping=True)
        '''

        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

        if True:
            rg_encoding = self.tokenizer(text_ori, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt")
        else:  
            rg_encoding = self.tokenizer(text, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt")
        # print(rg_encoding)
        '''
        self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        '''
        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }
    
    def get_label_similarity(self, vector1, vector2):
        intersection = torch.sum((vector1 == 1) & (vector2 == 1)).item()
        union = torch.sum((vector1 == 1) | (vector2 == 1)).item()
        
        if union == 0:
            return 1
        jaccard_similarity = intersection / union
        return jaccard_similarity

    def get_label_similarity_2(self, vector1, vector2):
        intersection = torch.sum((vector1 == 0) & (vector2 == 0)).item()
        union = torch.sum((vector1 == 0) | (vector2 == 0)).item()
        
        if union == 0:
            return 1
        jaccard_similarity = intersection / union
        return jaccard_similarity

    def count_different_bits(self, vector1, vector2):
        # Ensure the vectors are of the same length
        assert vector1.size() == vector2.size(), "Vectors must be of the same length"
        
        # Calculate the number of different bits
        different_bits = torch.sum(vector1 != vector2).item()
        
        return -different_bits


    def get_suite(self, index):
        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
                
                if self.usage == "train":
                    key = list(self.text.keys())[index]

                    minidx = 5
                    minvalue = 1
                    #for i in range(5, 20):
                    #    cs = self.count_different_bits(self.label[key], self.label[self.retrieve_name[key.split(".")[0] + ".npzlocal"][i].split(".")[0] + ".nii.gz"])
                    #    if cs < minvalue:
                    #        minvalue = cs
                    #        minidx = i
                    #print(minvalue, minidx)

                    if random.random() <= 0.5:
                        for i in range(5, 20):
                            cs = self.get_label_similarity(self.label[key], self.label[self.retrieve_name[key.split(".")[0] + ".npzlocal"][i].split(".")[0] + ".nii.gz"])
                            if cs < minvalue:
                                minvalue = cs
                                minidx = i
                    else:
                        for i in range(5, 20):
                            cs = self.get_label_similarity_2(self.label[key], self.label[self.retrieve_name[key.split(".")[0] + ".npzlocal"][i].split(".")[0] + ".nii.gz"])
                            if cs < minvalue:
                                minvalue = cs
                                minidx = i
                    #print(minvalue, minidx)
                    # minidx = 3
                    #while minidx == index:
                    #    minidx = random.randint(0, 20000)
                    txt = self.get_text(list(self.text.keys()).index(self.retrieve_name[key.split(".")[0] + ".npzlocal"][minidx].split(".")[0] + ".nii.gz"))
                    #txt = self.get_text(minidx)
                else:
                    txt = self.get_text((index+5) % len(self.text.keys()))

                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update({k+"_rej":v for k,v in txt.items()})
            for i in range(self.draw_false_image): # 0
                ret.update(self.get_false_image(i, selected_index=index))
            for i in range(self.draw_false_text): # 0
                ret.update(self.get_false_text(i, selected_index=index))
            result = True
            #except Exception as e:
            #    print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #    index = random.randint(0, len(self.index_mapper) - 1)

        name = list(self.text.keys())[ret["img_index"]]
        name_list = name.split(".")[0].split("_")
        clip_memory = np.load(os.path.join(self.clip_feature_path, name_list[0] + "_" + name_list[1]+name_list[2] + ".npy"))
        ret["clip_memory"] = torch.tensor(clip_memory).unsqueeze(0)

        llama_feature = np.load(os.path.join(self.llama_feature_path, name_list[0] + "_" + name_list[1]+ "_" +name_list[2] +"_1"+ ".npz_text_feature.npy"))
        ret["llama_feature"] = llama_feature
        
        return ret
    
    def __getitem__(self, index):
        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        new_ret["text_masks"] = ret["rg_attn"]

        new_ret["text_ids_rej"] = ret["rg_encoding_rej"]
        new_ret["text_masks_rej"] = ret["rg_attn_rej"]

        new_ret["seg_index"] = torch.tensor(ret["seg_index"])
        new_ret["obver_label"] = ret["obver_label"].long()
        new_ret["prompt_ids"] = ret["prompt_ids"]

        final_seg = ret["final_seg"]

        all_encoding = []
        for seg in final_seg:
            all_encoding.append(self.tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings = torch.cat([l.input_ids for l in all_encoding])
        all_maps = torch.cat([l.attention_mask for l in all_encoding])
        #print(all_encodings.shape)
        new_ret["all_encodings"] = all_encodings
        new_ret["all_maps"] = all_maps
        new_ret["image"] = ret["image"]
        new_ret["image_cd"] = ret["image_cd"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text_ori"] = ret["text"][0]
        
        
        return new_ret


class CTRGDataset_RATE_hr_sim(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380
        self.target_size = [240, 480, 480]

        # Image Transformations
        # not implement yet

        if self.usage == "train":
            self.text_path = CTRATE_TRAIN_REPORTS
            self.text_embedding_path = TEXT_EMBEDDING_DIR
            self.image_path = CTRATE_TRAIN_IMAGES
            self.label_path = CTRATE_TRAIN_LABELS
            self.region_report_path = CTRATE_TRAIN_REGION_REPORT
            self.feature_path = FEATURE_PATH_TRAIN
            self.mask_fast_path = CTRATE_TRAIN_MASK_FAST

            self.json_path = JSON_PATH_TRAIN
        else:
            self.text_path = CTRATE_VALID_REPORTS
            self.text_embedding_path = TEXT_EMBEDDING_DIR
            self.image_path = CTRATE_VALID_IMAGES
            self.label_path = CTRATE_VALID_LABELS
            self.region_report_path = CTRATE_VALID_REGION_REPORT
            self.feature_path = FEATURE_PATH_VALID
            self.mask_fast_path = CTRATE_VALID_MASK_FAST

            self.json_path = JSON_PATH_VALID


        self.retrieve_name = {}

        self.text = {}
        self.label = {}
        self.text_embedding = []
        all_text = {}
        all_label = {}

        self.all_impression = {}
        self.all_clinic_info = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)
        

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_text[cline[0]] = cline[3]
            self.all_impression[cline[0]] = cline[4]
            self.all_clinic_info[cline[0]] = cline[1]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = 3000
        else:
            limit = 10000000

        iii = 0
        start_id = 0
        end_id = start_id + 800000
        print(len(list(all_text.keys())))
        for l in list(all_text.keys()):
            iii += 1
            
            if True:
                if iii < start_id:
                    continue
                if iii > end_id:
                    continue
            
            key = l
            # if len(all_text[l]) < 500:
             #    continue
            # store_path = "/jizhicfs/datalh/M3AE/img_feature/version_2393_e7_unmask"
            # file_name = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz").split("/")[-1]
            # if os.path.exists(os.path.join(store_path, file_name + "selected_patch" + ".npy")):
            #     continue
            
            image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")
            feapath = os.path.join(self.text_embedding_path, key.split(".")[0] + "_text_embedding.npy")
            
            if os.path.exists(image_path) and limit > 0 and os.path.exists(feapath):
                # if os.path.exists(os.path.join("/jizhicfs/datalh/M3AE/img_feature/version_2818", key.split(".")[0] + ".npz" + "selected_patch" + ".npy")):
                #     continue
                # if not os.path.exists(os.path.join(self.text_embedding_path, key.split(".")[0] + "_text_embedding.npy")):
                #     continue
                # print(os.path.join("/jizhicfs/datalh/M3AE/img_feature/version_2818", key.split(".")[0] + ".npz" + "selected_patch" + ".npy"))
                # iii += 1
                # if iii < 300:
                #     continue
                
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]
                self.text_embedding.append(np.load(os.path.join(self.text_embedding_path, key.split(".")[0] + "_text_embedding.npy")))
        self.text_embedding = np.array(self.text_embedding)


        print(len(self.text.keys()))

         # get region report
        column1 = []
        column2 = []
        column3 = []
        column4 = []
        with open(self.region_report_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # 读取第一行（列名）
            
            # 读取每一行数据
            for row in csv_reader:
                column1.append(row[0])
                column2.append(row[1])
                column3.append(row[2])
                #@ if self.usage != "train":
                column4.append(row[3])

        all_name = list(set(column1))
        all_region = list(set([l for l in column2 if l != "" and "/" not in l]))
        
        self.name_2_region = {}
        self.name_2_abnorm = {}

        for patient in all_name:
            self.name_2_region[patient] = {}
            self.name_2_abnorm[patient] = {}

        for iidx, patient in enumerate(column1):
            if column2[iidx] == "" or "/" in column2[iidx]:
                continue
            self.name_2_region[patient][column2[iidx]] = column3[iidx]
            # if self.usage != "train":
            self.name_2_abnorm[patient][column2[iidx]] = column4[iidx]


    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.text.keys())

    def get_raw_image(self, index, image_key="image"):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path

    def pad_or_crop(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。
        
        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。
        mode (str): 模式，'train' 表示训练模式，'test' 表示测试模式。
        
        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        if self.usage == 'train':
            # 随机裁剪或填充
            start_depth = np.random.randint(0, max(target_depth - input_depth, 1))
            start_height = np.random.randint(0, max(target_height - input_height, 1))
            start_width = np.random.randint(0, max(target_width - input_width, 1))
        else:
            # 中心裁剪或填充
            start_depth = max((target_depth - input_depth) // 2, 0)
            start_height = max((target_height - input_height) // 2, 0)
            start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]

        return output_tensor

    def augment_image(self, image_array):
        """
        对3D图像进行数据增强。
        
        参数:
        image_array (numpy.ndarray): 3D图像数据
        
        返回:
        augmented_image (numpy.ndarray): 增强后的图像
        """
        # 随机旋转
        angle = np.random.uniform(-10, 10)  # 随机选择旋转角度
        rotated_image = rotate(image_array, angle, axes=(1, 2), reshape=False)
        
        ## 随机缩放
        #zoom_factor = np.random.uniform(0.9, 1.1)  # 随机选择缩放因子
        #zoomed_image = zoom(rotated_image, (1, zoom_factor, zoom_factor))
        
        return rotated_image

    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor
    
    def pad_or_crop3(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        _, input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size
        target_size = [9, target_depth, target_height, target_width]

        # 初始化输出张量
        output_tensor = torch.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[:, start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[:, input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor
    
    def image_prepro(self, image):

        # maybe crop like this
        # image = self.pad_or_crop(self.augment_image(image), target_size = self.target_size)
        image = self.pad_or_crop2(image, target_size = self.target_size)
        image = image.transpose(0,2,1)
        image = torch.tensor(image).float() 

        return image

    def get_image_fea(self, key):

        image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "selected_patch.npy")
        all_feature = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "image_feature.npy")
        text_feature = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "local_text.npy")
        img_patch = np.load(image_path)
        return img_patch, image_path, np.load(text_feature), np.load(all_feature)

    def get_mask(self, index, img_shape):
        key = list(self.text.keys())[index]
        mask_name = "_".join(key.split(".")[0].split("_")[:-1]) + "_1"
        fast_path = os.path.join(self.mask_fast_path, mask_name + ".npy")
        if not os.path.exists(fast_path):
            raise FileNotFoundError(
                f"Fast structural mask not found: {fast_path}. "
                "Fast masks ship with the data bundle under "
                "$CTRG_DATA_ROOT/ctrate/mask_fast/{train,valid}; check "
                "CTRG_DATA_ROOT / CTRATE_{TRAIN,VALID}_MASK_FAST."
            )
        return torch.tensor(np.load(fast_path))

    def get_image(self, index, image_key="image"):

        if False:
            key = list(self.text.keys())[index]
            image_tensor, path, text_feature, all_feature = self.get_image_fea(key)
            return {
                "image": torch.tensor(all_feature),
                "image_downsample": torch.tensor(all_feature), 
                "img_index": list(self.text.keys())[index],
                "cap_index": 0,
                "raw_index": index,
                "data_name": path.split("/")[-1],
                "obver_label": self.label[list(self.label.keys())[index]]
            }
        else:
            image, path = self.get_raw_image(index, image_key=image_key)
            mask = self.get_mask(index, image.shape)

            image = image*1000
            hu_min, hu_max = -1000, 200
            image = np.clip(image, hu_min, hu_max)

            image = (((image+400 ) / 600)).astype(np.float32)

            image_tensor = self.image_prepro(image)
            

            # image_downsample = F.interpolate(image_tensor.unsqueeze(0).unsqueeze(0), scale_factor=[192/240, 256/480, 256/480], mode='nearest').squeeze()

            return {
                "image": image_tensor,
                "image_downsample": mask, 
                "mask": mask,
                "img_index": list(self.text.keys())[index],
                "cap_index": 0,
                "raw_index": index,
                "data_name": path.split("/")[-1],
                "obver_label": self.label[list(self.label.keys())[index]]
            }

    def seg_text(self, text):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        # The patient has a port catheter.
        #Focal nodular opacity with vascular enlargement is observed in the right lung middle lobe adjacent to the major fissure, in the left lung lower lobe basal segment and lower lobe superior segment, 
        #and in the right lung lower lobe mediobasal segment, and it is suspicious for ultra-early Covid-19 pneumonia
        ###### pre seg
        
        segs = sent_tokenize(text)

        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        

        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == [] or idx_label[iidx] == [9]:

                if segs[iidx].lower().strip().startswith("the nodules") or segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]

                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx])
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other

        
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += ("No abnormalities in other organs.")
            
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        # print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def extract_all_abnormalities_grouped(self, data):
        organ_classes = [
            "trachea and bronchie", "mediastinum", "heart", "lung", "esophagus",
            "pleura", "bone", "thyroid", "breast", "abdomen", "others"
        ]
        output_classes = [
            "trachea and bronchie", "mediastinum and heart", "lung", "esophagus",
            "pleura", "bone", "thyroid", "breast", "abdomen", "others"
        ]
        temp_results = {k: [] for k in organ_classes}
        final_results = {k: "" for k in output_classes}

        def traverse(node, top_class, node_name=None):
            # 收集本节点的异常
            current_abn = set(node.get("attributes", {}).get("abnormality", [])) if isinstance(node, dict) else set()
            # 收集所有子节点的异常
            children_abn = set()
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "attributes":
                        continue
                    child_abn = traverse(v, top_class, k)
                    children_abn.update(child_abn)
            # 只保留本节点中未被子节点覆盖的异常
            unique_abn = current_abn - children_abn
            if unique_abn:
                name = node_name if node_name else top_class
                sentence = f"{name}: {', '.join(unique_abn)}"
                temp_results[top_class].append(sentence)
            # 返回本节点及所有子节点的异常合集
            return current_abn | children_abn

        if "hierarchical_abnormalities" in data:
            for organ, subnode in data["hierarchical_abnormalities"].items():
                if organ in organ_classes:
                    traverse(subnode, organ, organ)
                else:
                    traverse(subnode, "others", organ)

        # 合并mediastinum和heart
        temp_results["mediastinum and heart"] = temp_results["mediastinum"] + temp_results["heart"]

        # 输出整理
        for k in output_classes:
            if temp_results[k]:
                final_results[k] = "; ".join(temp_results[k])
            else:
                final_results[k] = f"{k}: No abnormality."

        return final_results

    def get_seg_text(self, patient_name):
        
        if True:
            # cregion_report = self.name_2_region[patient_name]
            cregion_report = self.name_2_abnorm[patient_name]

            if 'mediastinum' in cregion_report.keys():
                if "heart" in cregion_report.keys():
                    cregion_report["heart"] += cregion_report["mediastinum"]
                else:
                    cregion_report["heart"] = cregion_report["mediastinum"]
        
            regions = ['trachea and bronchie', 'heart', 'lung', 'esophagus', 'pleura',  'bone', 'thyroid', 'abdomen', 'breast', 'others']

            for r in regions:
                if r not in cregion_report.keys():
                    cregion_report[r] = r + ": No abnormality found." 
                elif bool(re.fullmatch(r'[\[\]\s]*', cregion_report[r])):
                    cregion_report[r] = r + ": No abnormality found." 
            
            final_seg = [cregion_report[r] for r in regions]
        else:
            json_path_c = os.path.join(self.json_path, patient_name.split(".")[0] + ".json")
            with open(json_path_c, 'r', encoding='utf-8') as f:
                json_c = json.load(f)
            cregion_report = self.extract_all_abnormalities_grouped(json_c)

            final_seg = [cregion_report[r] for r in list(cregion_report.keys())]
        

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        #print(final_text)
        # print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def get_label_similarity(self, vector1, vector2):
        intersection = torch.sum((vector1 == 1) & (vector2 == 1)).item()
        union = torch.sum((vector1 == 1) | (vector2 == 1)).item()
        
        if union == 0:
            return 1
        jaccard_similarity = intersection / union
        return jaccard_similarity

    def extract_abnorm(self, jpath):
        with open(jpath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        organ_keys = ["trachea and bronchie", "mediastinum", "heart", "lung", "esophagus", "pleura", "bone", "thyroid", "breast", "abdomen", "others"]
        return_sentence = []
        return_sentence_ab = []
        for organ in organ_keys:
            csentence = ""
            if len(data["hierarchical_abnormalities"][organ]["attributes"]["abnormality"]) != 0:
                if len(csentence) == 0:
                    csentence = organ + ", "
                csentence += ", ".join(data["hierarchical_abnormalities"][organ]["attributes"]["abnormality"]) + "."
            ab_sentence = csentence
            if len(data["hierarchical_abnormalities"][organ]["attributes"]["non_abnormality"]) != 0 and False:
                if len(csentence) == 0:
                    csentence = organ + ", "
                csentence += ", ".join(data["hierarchical_abnormalities"][organ]["attributes"]["non_abnormality"]) + "."
            if csentence != "":
                return_sentence.append(csentence)
            if ab_sentence != "":
                return_sentence_ab.append(ab_sentence)

            # print("all", csentence, "ab", ab_sentence)
        
        return_sentence.extend(data["global_abnormalities"])
        return_sentence.extend(data["impression_disorders"])

        return_sentence_ab.extend(data["global_abnormalities"])
        return_sentence_ab.extend(data["impression_disorders"])

        # final_abnorm_sentence = ". ".join(return_sentence) + "."
        final_abnorm_sentence = " ".join(return_sentence)
        final_abnorm_sentence_ab = " ".join(return_sentence_ab)
        # print(final_abnorm_sentence)
        return final_abnorm_sentence, final_abnorm_sentence_ab

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        key = list(self.text.keys())[raw_index]
        text_ori = self.text[list(self.text.keys())[raw_index]]

        items = key.split("_")
        if items[2] != "a":
            items[2] = chr(ord(items[2]) - 1)
            newkey = "_".join(items)
            _, last_text = self.extract_abnorm(os.path.join(self.json_path, newkey.split(".")[0] + ".json"))
        else:
            last_text = "Not given."

        clinical = self.all_clinic_info[key]
        impression = self.all_impression[key]

        if len(text_ori.split(" ") ) > 240:
            text = " ".join(text_ori.split(" ")[:220]) + "."
        else:
            text = text_ori
        encoding, text, final_idx, final_seg = self.seg_text(text)
        # encoding, text, final_idx, final_seg = self.get_seg_text(key)
        
        if False:
            if (self.usage != "train" or np.random.rand() > 0.5 or True):
                encoding_i, text_i, final_idx_i, final_seg_i = self.seg_text("No abnormalities in other organs.") # impression)
            else:
                encoding_i, text_i, final_idx_i, final_seg_i = self.seg_text(impression)

        # encoding, text, final_idx, final_seg = self.get_seg_text(key)
 

        # prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids


        rg_encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt")

        # print(np.load(os.path.join(self.text_embedding_path, key.split(".")[0] + "_text_embedding.npy")).shape)

        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                # "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
                "text_embedding": self.text_embedding[raw_index],

                # "impression_seg": final_seg_i,
                'impression': impression,
                "clinical": clinical,
                "last_text": last_text,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                # "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
                "text_embedding": self.text_embedding[raw_index],

                # "impression_seg": final_seg_i,
                'impression': impression,
                "clinical": clinical,
                "last_text": last_text,
            }

    def get_suite(self, index):
        #try:
        ret = dict()
        ret.update(self.get_image(index))
        if not self.image_only:
            txt = self.get_text(index)
            ret.update({"replica": True if txt["cap_index"] > 0 else False})
            ret.update(txt)

            if self.usage == "train" and False:
                key = list(self.text.keys())[index]

                minidx = 5
                minvalue = 1
                for i in range(5, 20):
                    cs = self.get_label_similarity(self.label[key], self.label[self.retrieve_name[key.split(".")[0] + ".npzlocal"][i].split(".")[0] + ".nii.gz"])
                    if cs < minvalue:
                        minvalue = cs
                        minidx = i
                minidx = 5
                # print(minvalue, minidx)
                txt = self.get_text(list(self.text.keys()).index(self.retrieve_name[key.split(".")[0] + ".npzlocal"][minidx].split(".")[0] + ".nii.gz"))
            else:
                pass
                    # txt = self.get_text((index+5) % len(self.text.keys()))

                # ret.update({k+"_rej":v for k,v in txt.items()})

            #except Exception as e:
            #    print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #    index = random.randint(0, len(self.index_mapper) - 1)
        # clip_memory = np.load(os.path.join(self.clip_feature_path, name_list[0] + "_" + name_list[1]+name_list[2] + ".npy"))
        # ret["clip_memory"] = torch.tensor(clip_memory).unsqueeze(0)

        # llama_feature = np.load(os.path.join(self.llama_feature_path, name_list[0] + "_" + name_list[1]+ "_" +name_list[2] +"_1"+ ".npz_text_feature.npy"))
        # ret["llama_feature"] = llama_feature
        
        return ret
    
    def __getitem__(self, index):
        # print(index)
        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        # new_ret["text_ids_rej"] = ret["rg_encoding_rej"]
        new_ret["text_masks"] = ret["rg_attn"]
        # new_ret["text_masks_rej"] = ret["rg_attn_rej"]
        new_ret["seg_index"] = torch.tensor(ret["seg_index"])
        new_ret["obver_label"] = ret["obver_label"].long()
        # new_ret["prompt_ids"] = ret["prompt_ids"]

        final_seg = ret["final_seg"]

        all_encoding = []
        for seg in final_seg:
            all_encoding.append(self.tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings = torch.cat([l.input_ids for l in all_encoding])
        all_maps = torch.cat([l.attention_mask for l in all_encoding])

        # tokenlizer impression
        if False:
            all_encoding = []
            final_seg_i = ret["impression_seg"]
            for seg in final_seg_i:
                all_encoding.append(self.tokenizer(seg, padding="max_length",
                    truncation=True,
                    max_length=150,
                    return_special_tokens_mask=True,
                    return_tensors="pt"))
            all_encodings_i = torch.cat([l.input_ids for l in all_encoding])
            all_maps_i = torch.cat([l.attention_mask for l in all_encoding])
            new_ret["all_encodings_i"] = all_encodings_i
            new_ret["all_maps_i"] = all_maps_i
            new_ret["impression_seg"] = torch.tensor([l[:13] == "No abnormalit" for l in final_seg_i]).unsqueeze(0)

        # print(new_ret["impression_seg"])

        #print(all_encodings.shape)
        new_ret["all_encodings"] = all_encodings
        new_ret["all_maps"] = all_maps
        new_ret["image"] = ret["image"]
        new_ret["image_downsample"] = ret["image_downsample"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text"] = ret["text"][0]
        new_ret["text_ori"] = ret["text"][0]
        # new_ret["text_ori"] = ret["impression"][0]

        # new_ret["text_rej"] = ret["text_rej"][0]
        new_ret["text_embedding"] = torch.tensor(ret["text_embedding"])

        new_ret["prior"] = "Clinical info: " + ret['clinical'] + " last report abnormality: " + ret['last_text']

        new_ret['impression'] = ret["impression"]
        
        new_ret['mask'] = ret["mask"]
        return new_ret
    
    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # print(dict_batch.keys())
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        for img_key in img_keys:
            # print(dict_batch[img_key][0].shape)
            cimage = torch.cat([l.unsqueeze(0) for l in dict_batch[img_key]], 0).unsqueeze(1)

            dict_batch[img_key] = cimage
        
        # exit(0)

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        
        if len(txt_keys) != 0:

            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = flatten_encodings
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                #mlm_ids, mlm_labels = (
                #    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                #    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                #)

                input_ids = []
                attention_mask = []
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids.append( _input_ids.unsqueeze(0))
                    attention_mask.append( _attention_mask.unsqueeze(0))

                input_ids = torch.cat(input_ids, 0)
                attention_mask = torch.cat(attention_mask, 0)

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                #dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                #dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"{txt_key}_ori"] = encoding
        
        return dict_batch
        

class CTRGDataset_RATE_hr(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380
        self.target_size = [240, 480, 480]

        # Image Transformations
        # not implement yet

        if self.usage == "train":
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/train_preprocessed_15_3"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_train_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_train"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_train"
            self.region_report_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_region_report.csv"
            self.json_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/report_hierarchy_tree/train"
        else:
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed_15_3"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_valid"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_valid"
            self.region_report_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radgenome_files_validation_region_report.csv"
            self.json_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/report_hierarchy_tree/valid"

        self.text = {}
        self.label = {}
        all_text = {}
        all_label = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_text[cline[0]] = cline[3]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = 300
        else:
            limit = 100000

        for l in list(all_text.keys()):
            key = l
            if len(all_text[l]) < 500:
                continue
            image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")
            if os.path.exists(image_path) and limit > 0:
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]

        print(len(self.text.keys()))

        # get region report
        column1 = []
        column2 = []
        column3 = []
        with open('/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radgenome_files_validation_region_report.csv', mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # 读取第一行（列名）
            
            # 读取每一行数据
            for row in csv_reader:
                column1.append(row[0])
                column2.append(row[1].split("/")[0])
                column3.append(row[2])

        all_name = list(set(column1))
        all_region = list(set([l for l in column2 if l != "" and "/" not in l]))
        
        self.name_2_region = {}

        for patient in all_name:
            self.name_2_region[patient] = {}

        for iidx, patient in enumerate(column1):
            if column2[iidx] == "" or "/" in column2[iidx]:
                continue
            self.name_2_region[patient][column2[iidx]] = column3[iidx]


    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.text.keys())

    def get_raw_image(self, index, image_key="image"):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path

    def pad_or_crop(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。
        
        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。
        mode (str): 模式，'train' 表示训练模式，'test' 表示测试模式。
        
        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        if self.usage == 'train':
            # 随机裁剪或填充
            start_depth = np.random.randint(0, max(target_depth - input_depth, 1))
            start_height = np.random.randint(0, max(target_height - input_height, 1))
            start_width = np.random.randint(0, max(target_width - input_width, 1))
        else:
            # 中心裁剪或填充
            start_depth = max((target_depth - input_depth) // 2, 0)
            start_height = max((target_height - input_height) // 2, 0)
            start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]

        return output_tensor

    def augment_image(self, image_array):
        """
        对3D图像进行数据增强。
        
        参数:
        image_array (numpy.ndarray): 3D图像数据
        
        返回:
        augmented_image (numpy.ndarray): 增强后的图像
        """
        # 随机旋转
        angle = np.random.uniform(-10, 10)  # 随机选择旋转角度
        rotated_image = rotate(image_array, angle, axes=(1, 2), reshape=False)
        
        ## 随机缩放
        #zoom_factor = np.random.uniform(0.9, 1.1)  # 随机选择缩放因子
        #zoomed_image = zoom(rotated_image, (1, zoom_factor, zoom_factor))
        
        return rotated_image

    
    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor
    

    def image_prepro(self, image):

        # maybe crop like this
        # image = self.pad_or_crop(self.augment_image(image), target_size = self.target_size)
        image = self.pad_or_crop2(image, target_size = self.target_size)
        image = image.transpose(0,2,1)
        image = torch.tensor(image).float() 

        return image

    def get_image(self, index, image_key="image"):
        image, path = self.get_raw_image(index, image_key=image_key)
        image_tensor = self.image_prepro(image)
        return {
            "image": image_tensor,
            "img_index": list(self.text.keys())[index],
            "cap_index": 0,
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.label[list(self.label.keys())[index]]
        }

    def seg_text(self, text):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        ###### pre seg
        segs = sent_tokenize(text)

        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == []:

                #print(xxx)
                # print(segs[iidx], segs[iidx-1].strip()[-1], iidx)
                #print(segs[iidx].strip()[:11])
                if segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]


                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]

                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += " " + ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += " " + ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += " " + ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += " " + ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += " " + ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += " " + ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += " " + ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += " " + ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += " " + ("No abnormalities in other organs.")
            
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        #print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
            )
        # print(final_seg)
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))

        #print(final_idx, final_token_merge)
        #for iidx in range(len(final_idx)-4):
        #    print(final_text[final_token_merge["offset_mapping"][final_idx[iidx]][0]:final_token_merge["offset_mapping"][final_idx[iidx+1]][0]])
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def extract_leaf_abnormalities_grouped(self, data):
        # 11个大类
        organ_classes = [
            "trachea and bronchie", "mediastinum", "heart", "lung", "esophagus",
            "pleura", "bone", "thyroid", "breast", "abdomen", "others"
        ]
        # 最终输出的10个大类
        output_classes = [
            "trachea and bronchie", "mediastinum and heart", "lung", "esophagus",
            "pleura", "bone", "thyroid", "breast", "abdomen", "others"
        ]
        # 初始化结果字典
        temp_results = {k: [] for k in organ_classes}
        final_results = {k: "" for k in output_classes}

        def is_leaf(node):
            return isinstance(node, dict) and set(node.keys()) == {"attributes"}

        def traverse(node, top_class):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "attributes":
                        continue
                    if is_leaf(v):
                        abn = v["attributes"].get("abnormality", [])
                        if abn:
                            # 只输出子节点和abnormality
                            sentence = f"{k}: {', '.join(abn)}"
                            temp_results[top_class].append(sentence)
                    else:
                        traverse(v, top_class)

        # 只处理hierarchical_abnormalities
        if "hierarchical_abnormalities" in data:
            for organ, subnode in data["hierarchical_abnormalities"].items():
                # 只处理在11个大类中的器官
                if organ in organ_classes:
                    traverse(subnode, organ)
                else:
                    traverse(subnode, "others")

        # 合并mediastinum和heart
        temp_results["mediastinum and heart"] = temp_results["mediastinum"] + temp_results["heart"]

        # 其余直接赋值
        for k in output_classes:
            if temp_results[k]:
                final_results[k] = "；".join(temp_results[k])
            else:
                final_results[k] = f"No abnormality in {k}"

        return final_results

    def get_seg_text(self, patient_name):
        
        if False:
            cregion_report = self.name_2_region[patient_name]

            if 'mediastinum' in cregion_report.keys():
                if "heart" in cregion_report.keys():
                    cregion_report["heart"] += cregion_report["mediastinum"]
                else:
                    cregion_report["heart"] = cregion_report["mediastinum"]
        
            regions = ['trachea and bronchie', 'heart', 'lung', 'esophagus', 'pleura',  'bone', 'thyroid', 'abdomen', 'breast', 'others']

            for r in regions:
                if r not in cregion_report.keys():
                    cregion_report[r] = "No abnormality found in " + r + "." 
        else:
            json_path_c = glob.glob(os.path.join(self.json_path, patient_name.split(".")[0] + ".json"))
            with open(json_path_c, 'r', encoding='utf-8') as f:
                json_c = json.load(f)
            cregion_report = self.extract_leaf_abnormalities_grouped(json_c)

        final_seg = [cregion_report[r] for r in list(cregion_report.keys())]

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        #print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
            )
        # print(final_seg)
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))

        #print(final_idx, final_token_merge)
        #for iidx in range(len(final_idx)-4):
        #    print(final_text[final_token_merge["offset_mapping"][final_idx[iidx]][0]:final_token_merge["offset_mapping"][final_idx[iidx+1]][0]])
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg


    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        text_ori = self.text[list(self.text.keys())[raw_index]]
        key = list(self.text.keys())[raw_index]
       
        if len(text_ori.split(" ") ) > 240:
            text = " ".join(text_ori.split(" ")[:220]) + "."
        else:
            text = text_ori
        # encoding, text, final_idx, final_seg = self.seg_text(text)
        encoding, text, final_idx, final_seg = self.get_seg_text(key)
 
        ''' for finetune
        encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                # return_tensors="pt",
                return_offsets_mapping=True)
        '''

        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

        if True:
            rg_encoding = self.tokenizer(text_ori, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                    return_offsets_mapping=True)
        else:  
            rg_encoding = self.tokenizer(text, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                    return_offsets_mapping=True)
        # print(rg_encoding)
        '''
        self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        '''
        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }

    def get_suite(self, index):
        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)

                if self.usage == "train":
                    key = list(self.text.keys())[index]

                    minidx = 5
                    minvalue = 1
                    for i in range(5, 20):
                        cs = self.get_label_similarity(self.label[key], self.label[self.retrieve_name[key.split(".")[0] + ".npzlocal"][i].split(".")[0] + ".nii.gz"])
                        if cs < minvalue:
                            minvalue = cs
                            minidx = i
                    # print(minvalue, minidx)
                    txt = self.get_text(list(self.text.keys()).index(self.retrieve_name[key.split(".")[0] + ".npzlocal"][minidx].split(".")[0] + ".nii.gz"))
                else:
                    txt = self.get_text((index+5) % len(self.text.keys()))

                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update({k+"_rej":v for k,v in txt.items()})


            for i in range(self.draw_false_image): # 0
                ret.update(self.get_false_image(i, selected_index=index))
            for i in range(self.draw_false_text): # 0
                ret.update(self.get_false_text(i, selected_index=index))
            result = True
            #except Exception as e:
            #    print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #    index = random.randint(0, len(self.index_mapper) - 1)

        name = list(self.text.keys())[ret["img_index"]]
        name_list = name.split(".")[0].split("_")
        clip_memory = np.load(os.path.join(self.clip_feature_path, name_list[0] + "_" + name_list[1]+name_list[2] + ".npy"))
        ret["clip_memory"] = torch.tensor(clip_memory).unsqueeze(0)

        llama_feature = np.load(os.path.join(self.llama_feature_path, name_list[0] + "_" + name_list[1]+ "_" +name_list[2] +"_1"+ ".npz_text_feature.npy"))
        ret["llama_feature"] = llama_feature
        
        return ret
    
    def __getitem__(self, index):

        #suite = self.get_suite(index)
        

        return self.get_suite(index)
 
    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # print(dict_batch.keys())
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        for img_key in img_keys:
            # print(dict_batch[img_key][0].shape)
            cimage = torch.cat([l.unsqueeze(0) for l in dict_batch[img_key]], 0).unsqueeze(1)

            dict_batch[img_key] = cimage
        
        # exit(0)

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        if len(txt_keys) != 0:
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.mlm_collator(flatten_encodings)
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"{txt_key}_ori"] = encoding

        return dict_batch


class CTRGDataset_RATE(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380
        self.target_size = [120, 256, 256]

        # Image Transformations
        # not implement yet

        if self.usage == "train":
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/train_preprocessed_15_3"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_train_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_train"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_train"
        else:
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed_15_3"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_valid"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_valid"

        self.text = {}
        self.label = {}
        all_text = {}
        all_label = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_text[cline[0]] = cline[3]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if cline[0].split("_")[-1][:1] != "1":
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = 100
        else:
            limit = 100000

        for l in list(all_text.keys()):
            key = l
            if len(all_text[l]) < 500:
                continue
            image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")
            if os.path.exists(image_path) and limit > 0:
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]

        print(len(self.text.keys()))

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.text.keys())

    def get_raw_image(self, index, image_key="image"):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path

    def pad_or_crop(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。
        
        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。
        mode (str): 模式，'train' 表示训练模式，'test' 表示测试模式。
        
        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        if self.usage == 'train':
            # 随机裁剪或填充
            start_depth = np.random.randint(0, max(target_depth - input_depth, 1))
            start_height = np.random.randint(0, max(target_height - input_height, 1))
            start_width = np.random.randint(0, max(target_width - input_width, 1))
        else:
            # 中心裁剪或填充
            start_depth = max((target_depth - input_depth) // 2, 0)
            start_height = max((target_height - input_height) // 2, 0)
            start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]

        return output_tensor

    def augment_image(self, image_array):
        """
        对3D图像进行数据增强。
        
        参数:
        image_array (numpy.ndarray): 3D图像数据
        
        返回:
        augmented_image (numpy.ndarray): 增强后的图像
        """
        # 随机旋转
        angle = np.random.uniform(-10, 10)  # 随机选择旋转角度
        rotated_image = rotate(image_array, angle, axes=(1, 2), reshape=False)
        
        ## 随机缩放
        #zoom_factor = np.random.uniform(0.9, 1.1)  # 随机选择缩放因子
        #zoomed_image = zoom(rotated_image, (1, zoom_factor, zoom_factor))
        
        return rotated_image

    
    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor
    

    def image_prepro(self, image):

        # maybe crop like this
        # image = self.pad_or_crop(self.augment_image(image), target_size = self.target_size)
        image = self.pad_or_crop2(image, target_size = self.target_size)
        image = image.transpose(0,2,1)
        image = torch.tensor(image).float() 

        return image

    def get_image(self, index, image_key="image"):
        image, path = self.get_raw_image(index, image_key=image_key)
        image_tensor = self.image_prepro(image)
        return {
            "image": image_tensor,
            "img_index": list(self.text.keys())[index],
            "cap_index": 0,
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.label[list(self.label.keys())[index]]
        }

    def seg_text(self, text):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        ###### pre seg
        segs = sent_tokenize(text)

        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == []:

                #print(xxx)
                # print(segs[iidx], segs[iidx-1].strip()[-1], iidx)
                #print(segs[iidx].strip()[:11])
                if segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]


                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]

                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        for iidx in range(1, len(idx_label)-1):
            
            if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
                idx_label[iidx] = [1]
            elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]):
                idx_label[iidx] = [2]
            elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
                idx_label[iidx] = [3]
            elif 0 in idx_label[iidx]:
                idx_label[iidx] = [0]
            

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx] + ",")
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += " " + ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += " " + ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += " " + ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += " " + ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += " " + ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += " " + ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += " " + ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += " " + ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += " " + ("No abnormalities in other organs.")
                continue
        
        # for iidx, seg in enumerate(final_seg):
        #     if len(seg) == 0:
        #         if iidx == 0:
        #             final_seg[iidx] += ("No abnormality found in trachea.")
        #         elif iidx == 1:
        #             final_seg[iidx] += ("No abnormality found in Mediastinum.")
        #         elif iidx == 2:
        #             final_seg[iidx] += " " + ("No abnormality found in heart.")
        #         elif iidx == 3:
        #             final_seg[iidx] += " " + ("No abnormality found in lung.")
        #         elif iidx == 4:
        #             final_seg[iidx] += " " + ("No abnormality found in esophagus.")
        #         elif iidx == 5:
        #             final_seg[iidx] += " " + ("No abnormality found in pleural.")
        #         elif iidx == 6:
        #             final_seg[iidx] += " " + ("No abnormalities in rib.")
        #         elif iidx == 7:
        #             final_seg[iidx] += " " + ("No abnormalities in chest.")
        #         elif iidx == 8:
        #             final_seg[iidx] += " " + ("No abnormalities in abdomen.")
        #         elif iidx > 8: # others
        #             final_seg[iidx] += " " + ("No abnormalities in other organs.")
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        #print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
            )
        # print(final_seg)
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))

        #print(final_idx, final_token_merge)
        #for iidx in range(len(final_idx)-4):
        #    print(final_text[final_token_merge["offset_mapping"][final_idx[iidx]][0]:final_token_merge["offset_mapping"][final_idx[iidx+1]][0]])
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        text_ori = self.text[list(self.text.keys())[raw_index]]

       
        if len(text_ori.split(" ") ) > 240:
            text = " ".join(text_ori.split(" ")[:220]) + "."
        else:
            text = text_ori
        encoding, text, final_idx, final_seg = self.seg_text(text)

 
        ''' for finetune
        encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                # return_tensors="pt",
                return_offsets_mapping=True)
        '''

        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

        if True:
            rg_encoding = self.tokenizer(text_ori, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                    return_offsets_mapping=True)
        else:  
            rg_encoding = self.tokenizer(text, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                    return_offsets_mapping=True)
        # print(rg_encoding)
        '''
        self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        '''
        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }

    def get_suite(self, index):
        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
            for i in range(self.draw_false_image): # 0
                ret.update(self.get_false_image(i, selected_index=index))
            for i in range(self.draw_false_text): # 0
                ret.update(self.get_false_text(i, selected_index=index))
            result = True
            #except Exception as e:
            #    print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #    index = random.randint(0, len(self.index_mapper) - 1)

        name = list(self.text.keys())[ret["img_index"]]
        name_list = name.split(".")[0].split("_")
        clip_memory = np.load(os.path.join(self.clip_feature_path, name_list[0] + "_" + name_list[1]+name_list[2] + ".npy"))
        ret["clip_memory"] = torch.tensor(clip_memory).unsqueeze(0)

        llama_feature = np.load(os.path.join(self.llama_feature_path, name_list[0] + "_" + name_list[1]+ "_" +name_list[2] +"_1"+ ".npz_text_feature.npy"))
        ret["llama_feature"] = llama_feature
        
        return ret
    
    def __getitem__(self, index):

        #suite = self.get_suite(index)
        

        return self.get_suite(index)
 
    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # print(dict_batch.keys())
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        for img_key in img_keys:
            # print(dict_batch[img_key][0].shape)
            cimage = torch.cat([l.unsqueeze(0) for l in dict_batch[img_key]], 0).unsqueeze(1)

            dict_batch[img_key] = cimage
        
        # exit(0)

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        if len(txt_keys) != 0:
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.mlm_collator(flatten_encodings)
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"{txt_key}_ori"] = encoding

        return dict_batch

def get_results(file_path):
    ff = open(file_path, "r")
    reader = csv.reader(ff)
    results = []
    reports = []
    for line in reader:
        if line[0] == "Report Impression":
            continue
        # print(line)
        cres = []
        for item in line[1:]:
            if item != "1.0":
                cres.append(0)
            else:
                cres.append(1)
        results.append(cres)
        reports.append(line[0])

    reports = {i:[l] for i, l in enumerate(reports)}    

    return results, reports

class CTRGDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380

        # Image Transformations
        # not implement yet

        textmodelpath = BIOMEDVLP_BERT
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = BertModel.from_pretrained(textmodelpath)
        self.tokenizer_text_model = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)
        
        # Read Texts
        self.text_path = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/label/chest/chest_report_english.txt"
        self.abnorma_path = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/label/chest/chest_abnormal_english.txt"
        self.image_path = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/chest_niigz_resize3"
        self.label_path = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/label/chest/chest_observation_label_8.txt"

        with open(self.text_path) as f:
            texts = eval(f.readline())['report']
        print(len(texts))

        with open(self.abnorma_path) as f:
            abnorms = eval(f.readline())['report']
        print(len(texts))

        obver_label = []
        with open(self.label_path) as f:
            line = f.readline()
            while line:
                obver_label.append(line)
                line = f.readline()
                
        print(len(obver_label))
        obver_list_all = []
        for obver in obver_label:
            obver_list_all.append([int(l) for l in list(obver[:8])])
        
        obver_list_all = np.array(obver_list_all)
        print(obver_list_all.shape)
        print(obver_list_all[:5])
        self.all_texts = texts

        path = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/label/chest/annotation.json"
        # 打开文件并读取内容  
        with open(path, 'r', encoding='utf-8') as file:  
            # 使用json.load()方法解析JSON数据  
            data = json.load(file) 

        print(len(data["train"]))
        
        self.index_map = [int(l["id"]) for l in data[self.usage]]

        if self.usage == "val":
            uu = "test"
        else:
            uu = self.usage
        
        self.index_map = [int(l["id"]) for l in data[uu]]

        if self.usage == "val":
            self.index_map = self.index_map[180:230]
        self.all_texts = [texts[l] for l in self.index_map]
        self.obver_list = [obver_list_all[l] for l in self.index_map]
        self.obver_list = torch.tensor(np.array(self.obver_list))
        self.abnorms_list = [abnorms[l] for l in self.index_map]

        if self.usage == "val":
            uu = "test"
            label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master-llm/val_text_ctrg_all.csv"
        else:
            uu = self.usage
            label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master-llm/train_text_ctrg_all.csv"

        
        self.label_list, _ = get_results(label_path)
        self.obver_list = [torch.tensor(l) for l in self.label_list]

        print(len(self.all_texts))

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_map)

    def get_raw_image(self, index, image_key="image"):
        
        image_path = os.path.join(self.image_path, "patient_" + str(self.index_map[index]) + ".nii.gz")
        return sitk.GetArrayFromImage(sitk.ReadImage(image_path)), image_path

    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor

    def image_prepro(self, image):

        # maybe crop like this
        image = self.pad_or_crop2(image[::2, :, :], target_size = (72, 192, 256))
        image = torch.tensor(image).float() 

        return (image - image.mean()) / image.std()

    def get_image(self, index, image_key="image"):
        image, path = self.get_raw_image(index, image_key=image_key)
        # print(image.shape)
        image_tensor = self.image_prepro(image)
        return {
            "image": image_tensor,
            "img_index": self.index_map[index],
            "cap_index": 0,
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.obver_list[index]
        }

    def seg_text(self, text):

        follow_segtence = ["patchy calcification shadow can be seen locally", 
                        "patchy low density shadow and patchy calcification can be seen", 
                        "Dot like calcification shadow can be seen on scanning", 
                        "The density of fat was decreased", "calcified nodules on the left", 
                        "Internal fixation image of left humeral head", 
                        "No obvious ascites sign is found", "slightly low density shadows"]
        
        keywords = [["thorax", "thoracic", "trachea"], ["rib", "spine", "sternum"], 
                    ["lung", "pulmonary"], ["heart", "mediastinum", "cardiac", "ventricle"], 
                    ["pleura", "pleural", "membrane", "diaphragm"], ["liver", "intrahepatic", "hepatic"], 
                    ["kidney", "renal", "adrenal"], ["thyroid"], 
                    ["breast", "mammary", "chest", "armpits", "armpit", "axillary"],
                    ["cavity", "cervical", "collar", "clavicular", "groins", "humerus", "glottis", "gyrus","ear"], 
                    ["gastric", "gas", "spleen", "splenic", "splenomegaly", "pancreas", "retroperitoneum", "danyi", "gallbladder", "biliary", "bile", "bilirubin", "cholecystectomy", "Danjianei"],
                    ["vertebral", "T9", "T12", "vertebrae", "skull", "bone"], ["aorta", "esophagus", "lumen","abdomen", "colon", "mesenteric"]]
        keywords = [["trachea", "bronchie", "bronchi", "bronc", "tracheostomy", "tracheost"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", "CTO", "arteries", "LAD"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "lobe", "Hilar"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag"], 
                    ["Pleura","thorax", "thoracic", "membrane", "diaphragm"], 
                    ["rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hiatal hernia", "hepatosteatosis", "peritoneum", "retrocrural"], 
                    ["recommended", "examination", "foramina", "lumen", "pleuroparenchymal", "appropriate", "bladder", "Perivesical", "prostate", "catheter"]]
        keywords = [["trachea", "bronchie", "bronchi", "bronc", "tracheostomy", "tracheost"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", "CTO", "arteries", "LAD"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "lobe", "Hilar"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag"], 
                    ["Pleura","thorax", "thoracic", "membrane", "diaphragm"], 
                    ["rib", "spine", "sternum", "bone", "skull", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular", "T9", "T12"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia"],
                    ["Abdomen","gastric", "gas", "Abdominal", "Adrenal", "Colon", "Duodenum","mesenteric", "Pericholecytic", "retroperitoneum", "danyi", "gallbladder", "biliary", "bile", "bilirubin", "cholecystectomy", "Danjianei", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic", "splenomegaly", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hiatal hernia", "hepatosteatosis", "peritoneum", "retrocrural"], 
                    ["recommended", "examination", "foramina", "lumen", "pleuroparenchymal", "appropriate", "bladder", "Perivesical", "prostate", "catheter"]]
        
        ##### pre seg 
        ###### pre seg
        segs = sent_tokenize(text)

        segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [-1 for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx] = keyidx
                        matched = True
                        break
                if matched:
                    break
        
        you = False

        for iidx in range(len(idx_label)):

            if idx_label[iidx] == -1:
                if segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            break
                        elif idx_label[kkk] != -1:
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] == -1:
                final_seg[8] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
            if idx_label[iidx] < 8:
                if len(self.tokenizer(final_seg[idx_label[iidx]])["input_ids"]):
                    final_seg[idx_label[iidx]] += (" " + segs[iidx] + ",")
            else:
                final_seg[8] += (" " + segs[iidx])  # merge all other
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += " " + ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += " " + ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += " " + ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += " " + ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += " " + ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += " " + ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += " " + ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += " " + ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += " " + ("No abnormalities in other organs.")
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l

        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                # return_special_tokens_mask=True,
                # return_offsets_mapping=True,
            )
        
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
        return final_token_merge, final_text, final_idx, final_seg

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        text_ori = self.all_texts[raw_index]
        current_ab = self.abnorms_list[raw_index]
       
        if len(text_ori.split(" ") ) > 200:
            text = " ".join(text_ori.split(" ")[:180]) + "."
        else:
            text = text_ori

        impression = current_ab
        encoding, text, final_idx, final_seg = self.seg_text(text)
        encoding_i, text_i, final_idx_i, final_seg_i = self.seg_text(impression)
 
        ''' for finetune
        encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                # return_tensors="pt",
                return_offsets_mapping=True)
        '''

        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

        rg_encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
                #return_offsets_mapping=True
                )
        # print(rg_encoding)

        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,

                "impression_seg": final_seg_i, # final_seg_i,
                'impression': impression, # impression,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,

                "impression_seg": final_seg_i, # final_seg_i,
                'impression': impression, # impression,
            }

    def get_suite(self, index):
        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
            for i in range(self.draw_false_image): # 0
                ret.update(self.get_false_image(i, selected_index=index))
            for i in range(self.draw_false_text): # 0
                ret.update(self.get_false_text(i, selected_index=index))
            result = True
            #except Exception as e:
            #    print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #    index = random.randint(0, len(self.index_mapper) - 1)
        return ret
    
    # def __getitem__(self, index):
    #     return self.get_suite(index)

    def __getitem__(self, index):

        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        new_ret["text_masks"] = ret["rg_attn"]
        new_ret["seg_index"] = torch.tensor(ret["seg_index"])
        new_ret["obver_label"] = ret["obver_label"].long()
        new_ret["prompt_ids"] = ret["prompt_ids"]

        final_seg = ret["final_seg"]

        all_encoding = []
        for seg in final_seg:
            all_encoding.append(self.tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings = torch.cat([l.input_ids for l in all_encoding])
        all_maps = torch.cat([l.attention_mask for l in all_encoding])
        #print(all_encodings.shape)
        new_ret["all_encodings"] = all_encodings
        new_ret["all_maps"] = all_maps
        new_ret["image"] = ret["image"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text_ori"] = ret["text"][0]
        
        ############################# tokenlizer impression

        new_ret['impression'] = ret["impression"]

        # tokenlizer impression
        all_encoding = []
        final_seg_i = ret["impression_seg"]
        for seg in final_seg_i:
            all_encoding.append(self.tokenizer_text_model(seg, padding="max_length",
                truncation=True,
                max_length=150,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings_i = torch.cat([l.input_ids for l in all_encoding])
        all_maps_i = torch.cat([l.attention_mask for l in all_encoding])
        new_ret["all_encodings_i"] = all_encodings_i# .unsqueeze(0)
        new_ret["all_maps_i"] = all_maps_i# .unsqueeze(0)
        new_ret["impression_seg"] = final_seg_i # torch.tensor([l[:13] == "No abnormalit" for l in final_seg_i]).unsqueeze(0)
        new_ret["impression_seg_text"] = final_seg_i
            
        return new_ret
    
    
    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # print(dict_batch.keys())
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        for img_key in img_keys:
            # print(dict_batch[img_key][0].shape)
            cimage = torch.cat([l.unsqueeze(0) for l in dict_batch[img_key]], 0).unsqueeze(1)

            dict_batch[img_key] = cimage
        
        # exit(0)

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        if len(txt_keys) != 0:
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            #flatten_mlms = self.mlm_collator(flatten_encodings)
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                #mlm_ids, mlm_labels = (
                #    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                #    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                #)

                #input_ids = torch.zeros_like(mlm_ids)
                #attention_mask = torch.zeros_like(mlm_ids)
                #for _i, encoding in enumerate(encodings):
                #    _input_ids, _attention_mask = (
                #        torch.tensor(encoding["input_ids"]),
                #        torch.tensor(encoding["attention_mask"]),
                #    )
                #    input_ids[_i, : len(_input_ids)] = _input_ids
                #    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                #dict_batch[f"{txt_key}_ids"] = input_ids
                #dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                #dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                #dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                # dict_batch[f"{txt_key}_masks"] = attention_mask
                # dict_batch[f"{txt_key}_ori"] = encoding

        return dict_batch


class CTRGDataset_(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str="",
            transform_keys: list=[],
            image_size: int=0,
            names: list=[],
            split: str = "train",
            text_column_name: str = "",
            max_text_len: int = 150,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.usage = split
        self.max_length = 380

        # Image Transformations
        # not implement yet

        # Read Texts
        self.text_path = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/label/chest/chest_report_english.txt"
        self.image_path = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/chest_niigz_resize3"
        self.label_path = "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/label/chest/chest_observation_label_8.txt"

        with open(self.text_path) as f:
            texts = eval(f.readline())['report']
        print(len(texts))

        obver_label = []
        with open(self.label_path) as f:
            line = f.readline()
            while line:
                obver_label.append(line)
                line = f.readline()
                
        print(len(obver_label))
        self.obver_list = []
        for obver in obver_label:
            self.obver_list.append([int(l) for l in list(obver[:8])])
        
        self.obver_list = torch.tensor(np.array(self.obver_list))
        print(self.obver_list.shape)
        print(self.obver_list[:5])
        self.all_texts = texts
        self.index_map = list(range(len(self.all_texts)))

        alldatalen = len(self.all_texts)
        if False:
            if self.usage == "train":
                self.all_texts = self.all_texts[:int(alldatalen * 0.8)]
                self.index_map = self.index_map[:int(alldatalen * 0.8)]
                self.obver_list = self.obver_list[:int(alldatalen * 0.8)]
                
            else:
                self.all_texts = self.all_texts[int(alldatalen * 0.8):int(alldatalen * 0.85)]
                self.index_map = self.index_map[int(alldatalen * 0.8):int(alldatalen * 0.85)]
                self.obver_list = self.obver_list[int(alldatalen * 0.8):int(alldatalen * 0.85)]
        else:
            if self.usage == "train":
                self.all_texts = self.all_texts[:int(alldatalen * 0.6)]
                self.index_map = self.index_map[:int(alldatalen * 0.6)]
                self.obver_list = self.obver_list[:int(alldatalen * 0.6)]
                
            elif self.usage == "val":
                self.all_texts = self.all_texts[int(alldatalen * 0.6):int(alldatalen * 0.62)]
                self.index_map = self.index_map[int(alldatalen * 0.6):int(alldatalen * 0.62)]
                self.obver_list = self.obver_list[int(alldatalen * 0.6):int(alldatalen * 0.62)]

            elif self.usage == "test":
                self.all_texts = self.all_texts[int(alldatalen * 0.8):int(alldatalen * 1)]
                self.index_map = self.index_map[int(alldatalen * 0.8):int(alldatalen * 1)]
                self.obver_list = self.obver_list[int(alldatalen * 0.8):int(alldatalen * 1)]
        print(len(self.all_texts))

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_map)

    def get_raw_image(self, index, image_key="image"):
        
        image_path = os.path.join(self.image_path, "patient_" + str(self.index_map[index]) + ".nii.gz")
        return sitk.GetArrayFromImage(sitk.ReadImage(image_path)), image_path

    def image_prepro(self, image):

        # maybe crop like this
        image = image[::2, :, :] 
        image = torch.tensor(image).float() 

        return (image - image.mean()) / image.std()

    def get_image(self, index, image_key="image"):
        image, path = self.get_raw_image(index, image_key=image_key)
        image_tensor = self.image_prepro(image)
        return {
            "image": image_tensor,
            "img_index": self.index_map[index],
            "cap_index": 0,
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.obver_list[index, :]
        }

    def seg_text(self, text):

        follow_segtence = ["patchy calcification shadow can be seen locally", 
                        "patchy low density shadow and patchy calcification can be seen", 
                        "Dot like calcification shadow can be seen on scanning", 
                        "The density of fat was decreased", "calcified nodules on the left", 
                        "Internal fixation image of left humeral head", 
                        "No obvious ascites sign is found", "slightly low density shadows"]
        
        keywords = [["thorax", "thoracic", "trachea"], ["rib", "spine", "sternum"], 
                    ["lung", "pulmonary"], ["heart", "mediastinum", "cardiac", "ventricle"], 
                    ["pleura", "pleural", "membrane", "diaphragm"], ["liver", "intrahepatic", "hepatic"], 
                    ["kidney", "renal", "adrenal"], ["thyroid"], 
                    ["breast", "mammary", "chest", "armpits", "armpit", "axillary"],
                    ["cavity", "cervical", "collar", "clavicular", "groins", "humerus", "glottis", "gyrus","ear"], 
                    ["gastric", "gas", "spleen", "splenic", "splenomegaly", "pancreas", "retroperitoneum", "danyi", "gallbladder", "biliary", "bile", "bilirubin", "cholecystectomy", "Danjianei"],
                    ["vertebral", "T9", "T12", "vertebrae", "skull", "bone"], ["aorta", "esophagus", "lumen","abdomen", "colon", "mesenteric"]]
        keywords = [["trachea", "bronchie", "bronchi", "bronc", "tracheostomy", "tracheost"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", "CTO", "arteries", "LAD"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "lobe", "Hilar"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag"], 
                    ["Pleura","thorax", "thoracic", "membrane", "diaphragm"], 
                    ["rib", "spine", "sternum", "bone", "skull", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular", "T9", "T12"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia"],
                    ["Abdomen","gastric", "gas", "Abdominal", "Adrenal", "Colon", "Duodenum","mesenteric", "Pericholecytic", "retroperitoneum", "danyi", "gallbladder", "biliary", "bile", "bilirubin", "cholecystectomy", "Danjianei", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic", "splenomegaly", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hiatal hernia", "hepatosteatosis", "peritoneum", "retrocrural"], 
                    ["recommended", "examination", "foramina", "lumen", "pleuroparenchymal", "appropriate", "bladder", "Perivesical", "prostate", "catheter"]]
        ##### pre seg
        ###### pre seg
        segs = sent_tokenize(text)


        segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [-1 for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx] = keyidx
                        matched = True
                        break
                if matched:
                    break
        
        you = False

        for iidx in range(len(idx_label)):

            if idx_label[iidx] == -1:
                if segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            break
                        elif idx_label[kkk] != -1:
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        ## inte segs
        final_seg = ["" for l in range(9)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] == -1:
                final_seg[8] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
            if idx_label[iidx] < 8:
                if len(self.tokenizer(final_seg[idx_label[iidx]])["input_ids"]):
                    final_seg[idx_label[iidx]] += (" " + segs[iidx] + ",")
            else:
                final_seg[8] += (" " + segs[iidx])  # merge all other
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("Thorax is symmetrical.")
                elif iidx == 1:
                    final_seg[iidx] += " " + ("No abnormality found in ribs.")
                elif iidx == 2:
                    final_seg[iidx] += " " + ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += " " + ("No abnormality found in heart.")
                elif iidx == 4:
                    final_seg[iidx] += " " + ("No pleural effusion or pleural thickening.")
                elif iidx == 5:
                    final_seg[iidx] += " " + ("No obvisous abnormalities in liver.")
                elif iidx == 6:
                    final_seg[iidx] += " " + ("No obvisous abnormalities in kidney.")
                elif iidx == 7:
                    final_seg[iidx] += " " + ("No obvisous abnormalities in thyroid.")
                elif iidx >= 8: # others
                    final_seg[iidx] += " " + ("No obvisous abnormalities in other organs.")
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l

        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
            )
        
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))
        
        while len(final_idx) < 10:
            final_idx.append(self.max_length)
        return final_token_merge, final_text, final_idx

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        text_ori = self.all_texts[raw_index]

       
        if len(text_ori.split(" ") ) > 200:
            text = " ".join(text_ori.split(" ")[:180]) + "."
        else:
            text = text_ori
        encoding, text, final_idx = self.seg_text(text)
 
        ''' for finetune
        encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                # return_tensors="pt",
                return_offsets_mapping=True)
        '''

        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

        rg_encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
                return_offsets_mapping=True)
        # print(rg_encoding)
        '''
        self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        '''
        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
            }

    def get_suite(self, index):
        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
            for i in range(self.draw_false_image): # 0
                ret.update(self.get_false_image(i, selected_index=index))
            for i in range(self.draw_false_text): # 0
                ret.update(self.get_false_text(i, selected_index=index))
            result = True
            #except Exception as e:
            #    print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #    index = random.randint(0, len(self.index_mapper) - 1)
        return ret
    
    def __getitem__(self, index):
        return self.get_suite(index)

    
    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # print(dict_batch.keys())
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        
        for img_key in img_keys:
            # print(dict_batch[img_key][0].shape)
            dict_batch[img_key] = [dict_batch[img_key][0].unsqueeze(0).unsqueeze(0)]

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        if len(txt_keys) != 0:
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.mlm_collator(flatten_encodings)
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"{txt_key}_ori"] = encoding

        return dict_batch
        

if __name__ == "__main__":
    dataset = CTRGDataset()

    for item in dataset:
        image = item["image"]
        print(image.shape)
        break