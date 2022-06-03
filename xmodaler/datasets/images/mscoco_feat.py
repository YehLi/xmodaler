# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import copy
import pickle
import random
from tqdm import tqdm
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from .mscoco import MSCoCoDataset
from ..build import DATASETS_REGISTRY

__all__ = ["MSCoCoFeatDataset"]

@DATASETS_REGISTRY.register()
class MSCoCoFeatDataset:
    @configurable
    def __init__(
        self,
        max_seq_len: int,
        max_feat_num: int,
        sample_ids,
        file_paths
    ):
        self.max_feat_num = max_feat_num
        self.sample_ids = sample_ids
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = {
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            'sample_ids': cfg.DATALOADER.SAMPLE_IDS,
            "file_paths": cfg.DATALOADER.FILE_PATHS
        }
        return ret

    def load_data(self, cfg):
        datalist = []
        for sample_id, file_path in zip(self.sample_ids, self.file_paths):
            datalist.append({
                kfg.IDS: sample_id,
                'path': file_path
            })
        return datalist

    def __call__(self, dataset_dict):
        sample_id = dataset_dict[kfg.IDS]
        path = dataset_dict['path']

        content = read_np(path)
        att_feats = content['features'][0:self.max_feat_num].astype('float32')
        global_feat = content['g_feature']

        ret = { 
            kfg.IDS: sample_id, 
            kfg.ATT_FEATS: att_feats,
            kfg.GLOBAL_FEATS: global_feat
        }

        g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
        ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
        dict_as_tensor(ret)
        return ret
        
        
