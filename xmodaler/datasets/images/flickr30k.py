# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import json
import jsonlines
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats, read_np_bbox
from xmodaler.tokenization import BertTokenizer
from ..build import DATASETS_REGISTRY

__all__ = ["Flickr30kDataset"]

@DATASETS_REGISTRY.register()
class Flickr30kDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_folder: str,
        anno_file: str,
        feats_folder: str,
        max_feat_num: int,
        max_seq_len: int,
        tokenizer
    ):
        self.stage = stage
        self.anno_folder = anno_folder
        self.anno_file = anno_file
        self.feats_folder = feats_folder
        self.max_feat_num = max_feat_num
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "all_data_final_train_2014.jsonline"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "all_data_final_val_set0_2014.jsonline"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "all_data_final_test_set0_2014.jsonline")
        }
        ret = {
            "stage": stage,
            "anno_folder": cfg.DATALOADER.ANNO_FOLDER,
            "anno_file": ann_files[stage],
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "tokenizer": BertTokenizer.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME,
                do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE)
        }
        return ret

    def load_raw_data(self, cfg):
        datalist = []
        with jsonlines.open(self.anno_file) as reader:
            for annotation in reader:
                sentences = annotation["sentences"]
                image_id = annotation["img_path"].split(".")[0]
                if self.stage == "train":
                    for sent in sentences:
                        datalist.append({ "image_id": image_id, "captions": sent })
                else:
                    datalist.append({ "image_id": image_id, "captions": sentences })
        return datalist

    def load_data(self, cfg):
        cache_path = os.path.join(
            self.anno_folder, "cache", 
            "RetrievalFlickr30k_%s_%d.pkl" % (self.stage, self.max_seq_len)
        )
        if not os.path.exists(cache_path):
            datalist = self.load_raw_data(cfg)    
            self.tokenize(datalist)
            pickle.dump(datalist, open(cache_path, "wb"))
        datalist = pickle.load(open(cache_path, "rb"))
        return datalist 
    
    def tokenize(self, datalist):
        for entry in datalist:
            captions = entry["captions"]

            if isinstance(captions, list):
                tokens_arr = []
                for caption in captions:
                    tokens = self.tokenizer.encode(caption)
                    tokens = tokens[: self.max_seq_len - 2]
                    tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)
                    tokens_arr.append(tokens)
                entry["captions"] = tokens_arr
            else:
                tokens = self.tokenizer.encode(captions)
                tokens = tokens[: self.max_seq_len - 2]
                tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)
                entry["captions"] = tokens

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        
        image_path = os.path.join(self.feats_folder, image_id + ".npz")
        features, image_locations = read_np_bbox(image_path, self.max_feat_num)

        captions = dataset_dict['captions']
        if self.stage == "train":
            u_tokens_type = np.array([0] * len(captions)).astype(np.int64)
            captions = np.array(captions).astype(np.int64)
            ids = image_id
        else:
            ids = [image_id, [image_id] * len(captions)]
            u_tokens_type = [ np.array([0] * len(caption)).astype(np.int64) for caption in captions ]
            captions = [np.array(caption).astype(np.int64) for caption in captions]
            
        ret = {
            kfg.ATT_FEATS: features.astype('float32'),
            kfg.ATT_FEATS_LOC: image_locations.astype('float32'),
            kfg.U_TOKENS_IDS: captions,
            kfg.U_TOKENS_TYPE: u_tokens_type,
        }

        dict_as_tensor(ret)
        ret.update({ kfg.IDS: ids })
        return ret