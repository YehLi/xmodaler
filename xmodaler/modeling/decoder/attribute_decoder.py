# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.base_attention import BaseAttention
from .decoder import Decoder
from .build import DECODER_REGISTRY

__all__ = ["AttributeDecoder"]

@DECODER_REGISTRY.register()
class AttributeDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        token_embed_dim: int,
        visual_feat_dim: int,
        attribute_dim: int, 
        dropout: float
    ):
        super(AttributeDecoder, self).__init__()
        self.num_layers = 1
        self.hidden_size = hidden_size

        self.attribute_fc = nn.Linear(attribute_dim, hidden_size)
        self.vfeat_fc = nn.Linear(visual_feat_dim, hidden_size)

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "token_embed_dim": cfg.MODEL.TOKEN_EMBED.DIM,
            "visual_feat_dim": cfg.MODEL.VISUAL_EMBED.IN_DIM,
            "attribute_dim": cfg.MODEL.LSTMA.ATTRIBUTE_DIM,
            "dropout": cfg.MODEL.LSTMA.DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.LSTMA = CN()
        cfg.MODEL.LSTMA.ATTRIBUTE_DIM = 1000
        cfg.MODEL.LSTMA.DROPOUT = 0.5

    def preprocess(self, batched_inputs):
        attributes = batched_inputs[kfg.ATTRIBUTE]
        gv_feats = batched_inputs[kfg.GLOBAL_FEATS]

        init_states = self.init_states(attributes.shape[0])
        hidden_states = init_states[kfg.G_HIDDEN_STATES]
        cell_states = init_states[kfg.G_CELL_STATES]
        
        # t = -2
        p_attributes = self.attribute_fc(attributes)
        if self.dropout is not None:
            p_attributes = self.dropout(p_attributes)
        h1_a, c1_a = self.lstm(p_attributes, (hidden_states[0], cell_states[0]))
        
        # t = -1
        p_gv_feats = self.vfeat_fc(gv_feats)
        if self.dropout is not None:
            p_gv_feats = self.dropout(p_gv_feats)
        h1_v, c1_v = self.lstm(p_gv_feats, (h1_a, c1_a))
        
        batched_inputs.update({ 
            kfg.G_HIDDEN_STATES: [h1_v],
            kfg.G_CELL_STATES: [c1_v]
        })
        return batched_inputs

    def forward(self, batched_inputs):
        wt = batched_inputs[kfg.G_TOKEN_EMBED]
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
        cell_states = batched_inputs[kfg.G_CELL_STATES]

        if self.dropout is not None:
            wt = self.dropout(wt)
        h1_t, c1_t = self.lstm(wt, (hidden_states[0], cell_states[0]))

        hidden_states = [h1_t]
        cell_states = [c1_t]
        return { 
            kfg.G_HIDDEN_STATES: hidden_states,
            kfg.G_CELL_STATES: cell_states
        }
