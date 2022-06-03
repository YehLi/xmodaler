# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import copy
import numpy as np
import weakref
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from xmodaler.utils import comm
from ..decode_strategy import build_beam_searcher, build_greedy_decoder

class Ensemble(nn.Module):
    def __init__(self, models, cfg):
        super(Ensemble, self).__init__()
        self.models = models
        self.num = len(self.models)
        self.weights = cfg.MODEL.MODEL_WEIGHTS
        self.beam_searcher = build_beam_searcher(cfg)

    def eval(self):
        for i in range(self.num):
            self.models[i].eval()

    def get_extended_attention_mask(self, batched_inputs):
        return comm.unwrap_model(self.models[0]).get_extended_attention_mask(batched_inputs)

    def preprocess_batch(self, batched_inputs):
        return comm.unwrap_model(self.models[0]).preprocess_batch(batched_inputs)

    def forward(self, batched_inputs, use_beam_search=None, output_sents=False):
        assert self.beam_searcher.beam_size > 1
        return self.beam_searcher(
            batched_inputs, 
            output_sents,
            model=weakref.proxy(self),
            weights = self.weights
        )