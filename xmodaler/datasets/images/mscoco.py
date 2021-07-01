import os
import copy
import pickle
import random
import numpy as np
import torch

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor
from ..build import DATASETS_REGISTRY

__all__ = ["MSCoCoDataset"]

@DATASETS_REGISTRY.register()
class MSCoCoDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        feats_folder: str
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_test.pkl")
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "seq_per_img": cfg.DATALOADER.SEQ_PER_IMG,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER
        }
        return ret

    def load_data(self, cfg):
        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        return datalist
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        
        #image_path  = os.path.join(self.feats_folder, '100001.npz')
        image_path = os.path.join(self.feats_folder, image_id + '.npz')
        att_feats = read_np(image_path).astype('float32')

        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
           att_feats = att_feats[:self.max_feat_num, :]

        if self.stage != "train":
            ret = { kfg.IDS: image_id, kfg.ATT_FEATS: att_feats }
            dict_as_tensor(ret)
            
        seq_len = len(dataset_dict['tokens_ids'][0,:])
        sent_num = len(dataset_dict['tokens_ids'])

        tokens_ids = np.zeros((self.seq_per_img, seq_len), dtype='int')
        target_ids = np.zeros((self.seq_per_img, seq_len), dtype='int')
  
        if sent_num >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(sent_num), self.seq_per_img)                
        else:
            sid = sent_num
            ixs = random.sample(range(sent_num), self.seq_per_img - sent_num)
            tokens_ids[0:sent_num, :] = dataset_dict['tokens_ids']
            target_ids[0:sent_num, :] = dataset_dict['target_ids']
           
        for i, ix in enumerate(ixs):
            tokens_ids[sid + i] = dataset_dict['tokens_ids'][ix,:]
            target_ids[sid + i] = dataset_dict['target_ids'][ix,:]

        ret = {
            kfg.IDS: image_id,
            kfg.TOKENS_IDS: tokens_ids,
            kfg.TARGET_IDS: target_ids,
            kfg.ATT_FEATS: att_feats
        }
        dict_as_tensor(ret)
        return ret
