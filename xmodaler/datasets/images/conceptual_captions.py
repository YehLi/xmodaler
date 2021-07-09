import os
import csv
import copy
import pickle
import random
from xmodaler.functional.func_pretrain import caption_to_mask_tokens
import numpy as np
import torch

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.tokenization import BertTokenizer
from xmodaler.functional import (
    read_lines_set, 
    read_np, 
    boxes_to_locfeats, 
    iou,
    random_word,
    random_region,
    dict_as_tensor
)

from ..build import DATASETS_REGISTRY

__all__ = ["ConceptualCaptionsDataset"]

@DATASETS_REGISTRY.register()
class ConceptualCaptionsDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        max_seq_length: int,
        max_feat_num: int,
        feats_folder: str,
        images_ids_file: str,
        tokenizer
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.max_seq_length = max_seq_length
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        self.images_ids_file =images_ids_file
        self.tokenizer = tokenizer
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "Train_GCC-training.tsv"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "Validation_GCC-1.1.0-Validation.tsv"),
        }
        images_ids_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "train_images_ids.txt"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "val_images_ids.txt")
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "max_seq_length": cfg.MODEL.MAX_SEQ_LEN,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "images_ids_file": images_ids_files[stage],
            "tokenizer": BertTokenizer.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME,
                do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE),
        }
        return ret

    def load_data(self, cfg):
        images_ids_set = read_lines_set(self.images_ids_file)

        datalist = []
        csv_rd = csv.reader(open(self.anno_file), delimiter='\t', quotechar='"')
        for imgid, row in enumerate(csv_rd):
            imgid_str = str(imgid + 1)
            if imgid_str in images_ids_set:
                datalist.append({
                    "image_id": imgid_str,
                    "caption": row[0]
                })  
        return datalist
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        caption = dataset_dict["caption"]
        image_path = os.path.join(self.feats_folder, image_id + ".npz")

        content = read_np(image_path)

        features = content['features'][0:self.max_feat_num - 1]
        cls_probs = content['cls_prob'][0:self.max_feat_num - 1]
        boxes = content['boxes'][0:self.max_feat_num - 1]
        image_h = content['image_h'][0]
        image_w = content['image_w'][0]
        num_boxes = len(boxes)
        
        image_locations = boxes_to_locfeats(boxes, image_w, image_h)
        overlaps = iou(boxes, boxes)

        tokens_ids, u_tokens_labels, g_tokens_labels = caption_to_mask_tokens(caption, self.max_seq_length, self.tokenizer)
        tokens_length = tokens_ids.shape[0]
        u_tokens_type = np.array([0] * tokens_length)
        g_tokens_type = np.array([1] * tokens_length)

        imgfeats, imgfeats_labels, masked_num = random_region(features, overlaps)
        imgfeats_labels = np.array(imgfeats_labels)
        valid_feats_num = max(1, num_boxes - masked_num)
        g_image_feat = np.sum(imgfeats, axis=0) / valid_feats_num
        g_image_location = np.array([0, 0, 1, 1, 1])

        imgfeats = np.concatenate([np.expand_dims(g_image_feat, axis=0), imgfeats], axis=0)
        image_locations = np.concatenate([np.expand_dims(g_image_location, axis=0), image_locations], axis=0)
        
        ret = {
            kfg.IDS: image_id,
            kfg.ATT_FEATS: imgfeats.astype('float32'),
            kfg.ATT_FEATS_LOC: image_locations.astype('float32'),
            kfg.U_TOKENS_TYPE: u_tokens_type.astype(np.int64),
            kfg.G_TOKENS_TYPE: g_tokens_type.astype(np.int64),
            kfg.U_TOKENS_IDS: tokens_ids.astype(np.int64),
            kfg.G_TOKENS_IDS: tokens_ids.astype(np.int64),
            kfg.U_TARGET_IDS: u_tokens_labels.astype(np.int64),
            kfg.G_TARGET_IDS: g_tokens_labels.astype(np.int64),
            kfg.V_TARGET: cls_probs.astype('float32'),
            kfg.V_TARGET_LABELS: imgfeats_labels.astype(np.int64)
        }

        dict_as_tensor(ret)
        return ret

