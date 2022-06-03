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
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from .mscoco import MSCoCoDataset
from ..build import DATASETS_REGISTRY
import clip
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip.clip import _convert_image_to_rgb
from timm.models.vision_transformer import resize_pos_embed

__all__ = ["MSCoCoRawDataset"]


def forward(self, x):
    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x
    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    attnpool = self.attnpool(x)
    return (x, attnpool)


@DATASETS_REGISTRY.register()
class MSCoCoRawDataset:
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
        model, preprocess = clip.load("RN101", device='cuda')
        forward_method = forward.__get__(model.visual, model.visual.__class__)
        setattr(model.visual, 'forward', forward_method)

        transform = Compose([
            Resize((448, 448), interpolation=Image.BICUBIC),
            CenterCrop((448, 448)),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        num_patches = 196
        pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, model.visual.attnpool.positional_embedding.size(-1), device='cuda'),)
        resized_pos_embed_weight = resize_pos_embed(model.visual.attnpool.positional_embedding.unsqueeze(0), pos_embed)
        pos_embed = nn.Parameter(resized_pos_embed_weight.squeeze(0),)
        model.visual.attnpool.positional_embedding = pos_embed

        self.model = model
        self.preprocess = transform
        self.pool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

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

        image = self.preprocess(Image.open(path)).unsqueeze(0).to('cuda')
        att_feats, global_feat = self.model.encode_image(image)
        att_feats = self.pool2d(att_feats)
        att_feats = att_feats.permute(0, 2, 3, 1)
        att_feats = att_feats.reshape(-1, att_feats.shape[-1])
        att_feats = att_feats[0:self.max_feat_num]

        ret = { 
            kfg.IDS: sample_id, 
            kfg.ATT_FEATS: att_feats.data.cpu().float().numpy(),
            kfg.GLOBAL_FEATS: global_feat.data.cpu().float().numpy()
        }

        g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
        ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
        dict_as_tensor(ret)
        return ret
        
        
