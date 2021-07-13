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
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats, read_np_bbox
from xmodaler.tokenization import BertTokenizer
from ..build import DATASETS_REGISTRY

__all__ = ["Flickr30k"]

@DATASETS_REGISTRY.register()
class Flickr30k:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_folder: str,
        ans2label_path: str,
        label2ans_path: str,
        feats_folder: str,
        max_feat_num: int,
        max_seq_len: int,
        tokenizer
    ):
        pass

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        pass