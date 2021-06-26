import os
import copy
import pickle
import random
import numpy as np
import torch

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..build import DATASETS_REGISTRY

__all__ = ["MSCoCoDatasetMapper"]

@DATASETS_REGISTRY.register()
class MSCoCoDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train: bool,
        seq_per_img: int,
        max_feat_num: int,
        feats_folder: str
    ):
        self.is_train = is_train
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = {
            "is_train": is_train,
            "seq_per_img": cfg.DATALOADER.SEQ_PER_IMG,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER
        }
        return ret

    def load_data(self, cfg, stage):
        anno_file = cfg.DATALOADER.ANNO_FILE + '_' + stage + '.pkl'
        datalist = pickle.load(open(anno_file, 'rb'), encoding='bytes')
        return datalist
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        
        att_feats = np.load(os.path.join(self.feats_folder, image_id + '.npz'))['feat']
        #att_feats = np.load(os.path.join(self.feats_folder, '100001' + '.npz'))['feat']
        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
           att_feats = att_feats[:self.max_feat_num, :]
           
        att_feats = torch.as_tensor(np.array(att_feats).astype('float32'))

        if not self.is_train:
            return { kfg.IDS: image_id, kfg.ATT_FEATS: att_feats }

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

        tokens_ids = torch.as_tensor(tokens_ids)
        target_ids = torch.as_tensor(target_ids)

        return {
            kfg.IDS: image_id,
            kfg.TOKENS_IDS: tokens_ids,
            kfg.TARGET_IDS: target_ids,
            kfg.ATT_FEATS: att_feats
        }
