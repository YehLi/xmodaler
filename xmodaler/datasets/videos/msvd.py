# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jingwen Chen
@contact: yehaoli.sysu@gmail.com, chenjingwen.sysu@gmail.com
"""
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

__all__ = ["MSVDDataset"]

@DATASETS_REGISTRY.register()
class MSVDDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        feats_folder: str
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        self.max_seq_len = max_seq_len

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msvd_caption_anno_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msvd_caption_anno_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "msvd_caption_anno_test.pkl")
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN
        }
        return ret

    def load_data(self, cfg):
        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        if self.stage == 'train':
            expand_datalist = []
            for data in datalist:
                for token_id, target_id in zip(data['tokens_ids'], data['target_ids']):
                    expand_datalist.append({
                        'video_id': data['video_id'],
                        'tokens_ids': np.expand_dims(token_id, axis=0),
                        'target_ids': np.expand_dims(target_id, axis=0)
                    })
            datalist = expand_datalist
        return datalist
        
    def _sample_frame(self, atten_feats):
        interval = atten_feats.shape[0] / self.max_feat_num
        selected_indexes = [int(i * interval) for i in range(self.max_feat_num)]
        selected_frames = atten_feats[selected_indexes, :]
        return selected_frames

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        video_id = dataset_dict['video_id']

        feat_path  = os.path.join(self.feats_folder, video_id + '.npy')
        content = read_np(feat_path)
        att_feats = content['features'].astype('float32')
        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
            att_feats = self._sample_frame(att_feats)
            assert att_feats.shape[0] == self.max_feat_num

        ret = { kfg.IDS: video_id, kfg.ATT_FEATS: att_feats }

        if self.stage != 'train':
            g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
            dict_as_tensor(ret)
            return ret

        sent_num = len(dataset_dict['tokens_ids'])
        if sent_num >= self.seq_per_img:
            selects = random.sample(range(sent_num), self.seq_per_img)
        else:
            selects = random.choices(range(sent_num), k = (self.seq_per_img - sent_num))
            selects += list(range(sent_num))

        tokens_ids = [ dataset_dict['tokens_ids'][i,:].astype(np.int64) for i in selects ]
        target_ids = [ dataset_dict['target_ids'][i,:].astype(np.int64) for i in selects ]
        g_tokens_type = [ np.ones((len(dataset_dict['tokens_ids'][i,:]), ), dtype=np.int64) for i in selects ]
        
        ret.update({
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
        })
        dict_as_tensor(ret)
        return ret
        