# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer
from .build import ENCODER_REGISTRY

__all__ = ["GTransformerEncoder"]

@ENCODER_REGISTRY.register()
class GTransformerEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        bert_layers,
    ):
        super(GTransformerEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers
        self.gvfeat_embed = nn.Sequential(
            nn.Linear(hidden_size * (num_hidden_layers + 1), hidden_size),
            torch.nn.LayerNorm(hidden_size)
        )

    @classmethod
    def from_config(cls, cfg):
        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "bert_layers": bert_layers,
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.COSNET = CN()
        cfg.MODEL.COSNET.NUM_CLASSES = 906

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            vfeats = batched_inputs[kfg.ATT_FEATS]
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
            ext_vmasks = torch.cat([ext_vmasks[:,:,:,0:1], ext_vmasks], dim=-1)
            ret.update({ kfg.EXT_ATT_MASKS: ext_vmasks })

            gfeats = []
            gfeats.append(vfeats[:, 0])
            for layer_module in self.layers:
                vfeats, _ = layer_module(vfeats, ext_vmasks)
                gfeats.append(vfeats[:, 0])
            gfeats = torch.cat(gfeats, dim=-1)
            gfeats = self.gvfeat_embed(gfeats)
            vfeats = torch.cat([gfeats.unsqueeze(1), vfeats[:, 1:]], dim=1)
            ret.update({ kfg.ATT_FEATS: vfeats })
        return ret