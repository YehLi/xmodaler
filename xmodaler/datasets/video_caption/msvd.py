import os
import copy
import pickle
import random
import numpy as np
import torch

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..build import DATASETS_REGISTRY

__all__ = ["MSVDDatasetMapper"]

@DATASETS_REGISTRY.register()
class MSVDDatasetMapper:
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
        data = pickle.load(open(anno_file, 'rb'), encoding='bytes')
        # datalist = [{"image_id": int, "tokens_ids": ndarray , "target_ids": ndarray}, ...]
        if self.is_train:
            return data
        else:
            datalist = []
            visited = set()
            for item in data:
                if item["image_id"] not in visited:
                    visited.add(item["image_id"])
                    datalist.append(item)
            return datalist
    
    def _sample_frame(self, atten_feats):
        while len(atten_feats) % self.max_feat_num > 0:
            atten_feats = np.concatenate([atten_feats, atten_feats[-1:, :]], axis=0)
        step = len(atten_feats) // self.max_feat_num
        return atten_feats[::step, :]

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        
        att_feats = np.load(os.path.join(self.feats_folder, "{}.npy".format(image_id)))
        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
            att_feats = self._sample_frame(att_feats)
            assert att_feats.shape[0] == self.max_feat_num
           
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
