# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import copy
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer
from .build import ENCODER_REGISTRY

__all__ = ["TransformerEncoder"]

@ENCODER_REGISTRY.register()
class TransformerEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        bert_layer: BertLayer,
    ):
        super(TransformerEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = nn.ModuleList(
            [copy.copy(bert_layer) for _ in range(self.num_hidden_layers)]
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "bert_layer": BertLayer(cfg),
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            vfeats = batched_inputs[kfg.ATT_FEATS]
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
            for layer_module in self.layers:
                vfeats, _ = layer_module(vfeats, ext_vmasks)
            ret.update({ kfg.ATT_FEATS: vfeats })
        return ret