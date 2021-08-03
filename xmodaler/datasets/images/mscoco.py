# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
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
        max_seq_len: int,
        feats_folder: str,
        relation_file: str
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        self.max_seq_len = max_seq_len
        self.relation_file = relation_file
        
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
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "relation_file": cfg.DATALOADER.RELATION_FILE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN
        }
        return ret

    def load_data(self, cfg):
        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        if len(self.relation_file) > 0:
            relation = pickle.load(open(self.relation_file, 'rb'), encoding='bytes')
            for i in range(len(datalist)):
                image_id = int(datalist[i]['image_id'])
                if image_id in relation:
                    datalist[i]['relation'] = relation[image_id]
        return datalist
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        
        #feat_path  = os.path.join(self.feats_folder, '100001.npz')
        feat_path = os.path.join(self.feats_folder, image_id + '.npz')
        content = read_np(feat_path)
        att_feats = content['features'][0:self.max_feat_num].astype('float32')
        ret = { kfg.IDS: image_id, kfg.ATT_FEATS: att_feats }

        if "boxes" in content:
            att_feats = att_feats[0:self.max_feat_num - 1]
            cls_probs = content['cls_prob'][0:self.max_feat_num - 1]
            boxes = content['boxes'][0:self.max_feat_num - 1]
            image_h = content['image_h'][0]
            image_w = content['image_w'][0]
            image_locations = boxes_to_locfeats(boxes, image_w, image_h)

            g_image_feat = np.mean(att_feats, axis=0)
            att_feats = np.concatenate([np.expand_dims(g_image_feat, axis=0), att_feats], axis=0)
            g_image_location = np.array([0, 0, 1, 1, 1])
            image_locations = np.concatenate([np.expand_dims(g_image_location, axis=0), image_locations], axis=0)

            ret.update({
                kfg.ATT_FEATS: att_feats,
                kfg.V_TARGET: cls_probs.astype('float32'),
                kfg.ATT_FEATS_LOC: image_locations.astype('float32'),
            })
        if 'relation' in dataset_dict:
            ret.update( { kfg.RELATION: dataset_dict['relation']} )
            
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
