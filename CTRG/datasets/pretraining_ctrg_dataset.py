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

        # Image Transformations
        # not implement yet
        #/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1823_e6
        #/jizhicfs/datalh/M3AE/img_feature/version_1859_e8
        #/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1859_e8
        # /apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1823_e6

        if self.usage == "train":
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/train_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_train_predicted_labels.csv"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_train"
            self.feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1687"
            self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/version_1859_e8"
            # self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/version_2264_e10"
            # self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/version_2288_e8"
        else:
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_valid"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_valid"
            self.feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/version_1687"
            # self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/version_1859_e8"
            # self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/version_2264_e10"
            # self.feature_path = "/jizhicfs/datalh/M3AE/img_feature/version_2288_e8"

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
            limit = 30
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
        segs = re.split(r'[.]', text)
        # print(len(text.split(" ")))

        segs = [l for l in segs if len(l) > 1]


        ## remove the split by decimal point

        have_digit = True
        while have_digit:
            tt = True
            for iidx, seg in enumerate(segs[1:]):
                if seg[0] != " " :    # if the first is not " ", there is a digit?
                    tt = False
                    break
            # print(iidx, len(segs))
            if iidx == len(segs)-2 and tt:
                have_digit = False
            else:
                # print(iidx, seg, segs[iidx])
                segs[iidx] += ("."+seg)
                segs.remove(seg)

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
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/train_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_train_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_train"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_train"
            self.retrieve_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/retravel1859_train.pkl"
            self.region_report_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_region_report.csv"
        else:
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_valid"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_valid"
            self.retrieve_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/retravel1859.pkl"
            self.region_report_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radgenome_files_validation_region_report.csv"


        with open(self.retrieve_path, 'rb') as f:
            self.retrieve_name = pickle.load(f)

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
        with open(self.region_report_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # 读取第一行（列名）
            
            # 读取每一行数据
            for row in csv_reader:
                column1.append(row[0])
                column2.append(row[1])
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

        '''
        self.new_text = {}
        for key in (self.text.keys()):
            patient_name = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz").split("/")[-1]
            if os.path.exists(os.path.join("/jizhicfs/datalh/M3AE/img_feature/version_2262_e6", patient_name + "selected_patch" + ".npy")):
                continue
            self.new_text[key] = self.text[key]
        self.text = self.new_text

        print(len(self.text.keys()))

        '''

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

        # print(image.shape)

        image = image*1000
        hu_min, hu_max = -1000, 200
        image = np.clip(image, hu_min, hu_max)

        image = (((image+400 ) / 600)).astype(np.float32)

        image_tensor = self.image_prepro(image)

        image_downsample = F.interpolate(image_tensor.unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze()

        return {
            "image": image_tensor,
            "image_downsample": image_downsample, 
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
        segs = re.split(r'[.]', text)
        # print(len(text.split(" ")))

        segs = [l for l in segs if len(l) > 1]


        ## remove the split by decimal point

        have_digit = True
        while have_digit:
            tt = True
            for iidx, seg in enumerate(segs[1:]):
                if seg[0] != " " :    # if the first is not " ", there is a digit?
                    tt = False
                    break
            # print(iidx, len(segs))
            if iidx == len(segs)-2 and tt:
                have_digit = False
            else:
                # print(iidx, seg, segs[iidx])
                segs[iidx] += ("."+seg)
                segs.remove(seg)

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

                #print(xxx)
                # print(segs[iidx], segs[iidx-1].strip()[-1], iidx)
                #print(segs[iidx].strip()[:11])
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

        

        #for iidx in range(1, len(idx_label)-1):
            
            #if 1 in idx_label[iidx] and (1 in idx_label[iidx-1] or 1 in idx_label[iidx+1]):
            #    idx_label[iidx] = [1]
            #elif 2 in idx_label[iidx] and (2 in idx_label[iidx-1] or 2 in idx_label[iidx+1]): # 气管跟肺部不做区分
            #    idx_label[iidx] = [2]
            #elif 3 in idx_label[iidx] and (3 in idx_label[iidx-1] or 3 in idx_label[iidx+1]):
            #    idx_label[iidx] = [3]
            #elif 0 in idx_label[iidx]:
            #    idx_label[iidx] = [0]
        

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

    def get_seg_text(self, patient_name):

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

        final_seg = [cregion_report[r] for r in regions]

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

    def get_label_similarity(self, vector1, vector2):
        intersection = torch.sum((vector1 == 1) & (vector2 == 1)).item()
        union = torch.sum((vector1 == 1) | (vector2 == 1)).item()
        
        if union == 0:
            return 1
        jaccard_similarity = intersection / union
        return jaccard_similarity



    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        key = list(self.text.keys())[raw_index]
        text_ori = self.text[list(self.text.keys())[raw_index]]

       
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
        print(index)
        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        new_ret["text_ids_rej"] = ret["rg_encoding_rej"]
        new_ret["text_masks"] = ret["rg_attn"]
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
        new_ret["image_downsample"] = ret["image_downsample"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text"] = ret["text"][0]

        new_ret["text_rej"] = ret["text_rej"][0]
        
        
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
        else:
            self.text_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv"
            self.image_path = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed_15_3"
            self.label_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
            self.clip_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/scripts/output_folder_valid"
            self.llama_feature_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/patch_selection/llm2vec_llama3_8b_valid"
            self.region_report_path = "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radgenome_files_validation_region_report.csv"

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
        segs = re.split(r'[.]', text)
        # print(len(text.split(" ")))

        segs = [l for l in segs if len(l) > 1]


        ## remove the split by decimal point

        have_digit = True
        while have_digit:
            tt = True
            for iidx, seg in enumerate(segs[1:]):
                if seg[0] != " " :    # if the first is not " ", there is a digit?
                    tt = False
                    break
            # print(iidx, len(segs))
            if iidx == len(segs)-2 and tt:
                have_digit = False
            else:
                # print(iidx, seg, segs[iidx])
                segs[iidx] += ("."+seg)
                segs.remove(seg)

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

    def get_seg_text(self, patient_name):

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
        segs = re.split(r'[.]', text)
        # print(len(text.split(" ")))

        segs = [l for l in segs if len(l) > 1]


        ## remove the split by decimal point

        have_digit = True
        while have_digit:
            tt = True
            for iidx, seg in enumerate(segs[1:]):
                if seg[0] != " " :    # if the first is not " ", there is a digit?
                    tt = False
                    break
            # print(iidx, len(segs))
            if iidx == len(segs)-2 and tt:
                have_digit = False
            else:
                # print(iidx, seg, segs[iidx])
                segs[iidx] += ("."+seg)
                segs.remove(seg)

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
        image = self.pad_or_crop2(image[::2, :, :], target_size = (64, 256, 256))
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
        segs = re.split(r'[,.;]', text)

        segs = [l for l in segs if len(l) > 1]

        have_digit = True
        while have_digit:
            tt = True
            for iidx, seg in enumerate(segs[1:]):
                if seg[0] != " " :    # if the first is not " ", there is a digit?
                    tt = False
                    break
            # print(iidx, len(segs))
            if iidx == len(segs)-2 and tt:
                have_digit = False
            else:
                # print(iidx, seg, segs[iidx])
                segs[iidx] += ("."+seg)
                segs.remove(seg)


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
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
        return final_token_merge, final_text, final_idx, final_seg

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        text_ori = self.all_texts[raw_index]

       
        if len(text_ori.split(" ") ) > 200:
            text = " ".join(text_ori.split(" ")[:180]) + "."
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

        rg_encoding = self.tokenizer(text_ori, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
                return_offsets_mapping=True)
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
        segs = re.split(r'[,.;]', text)

        segs = [l for l in segs if len(l) > 1]

        have_digit = True
        while have_digit:
            tt = True
            for iidx, seg in enumerate(segs[1:]):
                if seg[0] != " " :    # if the first is not " ", there is a digit?
                    tt = False
                    break
            # print(iidx, len(segs))
            if iidx == len(segs)-2 and tt:
                have_digit = False
            else:
                # print(iidx, seg, segs[iidx])
                segs[iidx] += ("."+seg)
                segs.remove(seg)


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
