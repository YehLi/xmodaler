# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.multihead_attention import MultiHeadAttention
from ..layers.positionwise_feedforward import PositionWiseFeedForward

from .decoder import Decoder
from .build import DECODER_REGISTRY

import numpy as np

__all__ = ["MeshedDecoder"]

class MeshedDecoderLayer(nn.Module):
    def __init__(
        self, 
        *,
        d_model=512, 
        num_head=8, 
        d_ff=2048, 
        dropout=.1,
        enc_layer_num=3,
    ):
        super(MeshedDecoderLayer, self).__init__()

        d_k = d_v = d_model // num_head

        self.self_att = MultiHeadAttention( d_model=d_model, 
                                            d_k=d_k, 
                                            d_v=d_v, 
                                            num_head=num_head, 
                                            dropout=dropout
                                        )
        self.enc_att = MultiHeadAttention(  d_model=d_model, 
                                            d_k=d_k, 
                                            d_v=d_v, 
                                            num_head=num_head, 
                                            dropout=dropout
                                        )
                                        
        self.pwff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.fc_alpha = nn.ModuleList()
        for _ in range(enc_layer_num):
            self.fc_alpha.append(nn.Linear(2 * d_model, d_model))

        # init fc_alpha weights
        for i in range(enc_layer_num):
            nn.init.xavier_uniform_(self.fc_alpha[i].weight)
            nn.init.constant_(self.fc_alpha[i].bias, 0)

    def forward(self, input, enc_output, mask_self_att, mask_enc_att, history_states=None):
        self_att = self.self_att(input, input, input, mask_self_att, history_states=history_states)

        # cal attention on each encoder layer then weighted sum
        enc_att = 0
        for i in range(len(self.fc_alpha)):
            enc_att_k = self.enc_att(self_att, keys=enc_output[:, i], values=enc_output[:, i], attention_mask=mask_enc_att)
            alpha_k = torch.sigmoid(self.fc_alpha[i](torch.cat([self_att, enc_att_k], -1)))
            enc_att += enc_att_k * alpha_k
        enc_att = enc_att / np.sqrt(len(self.fc_alpha))

        ff = self.pwff(enc_att)
        return ff

@DECODER_REGISTRY.register()
class MeshedDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        d_model: int , 
        num_layer: int,  
        num_att_head: int, 
        d_ff: int, 
        dropout: float,
        padding_idx: int, # -1
        enc_layer_num: int
    ):
        super(MeshedDecoder, self).__init__()

        self.num_layers = num_layer
        self.d_model = d_model
        self.num_att_head = num_att_head
        self.d_ff = d_ff
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.layers = nn.ModuleList([
            MeshedDecoderLayer(
                d_model=self.d_model, 
                num_head=self.num_att_head, 
                d_ff=self.d_ff, 
                dropout=self.dropout,
                enc_layer_num=enc_layer_num
            ) for _ in range(self.num_layers)
        ])

    @classmethod
    def from_config(cls, cfg):
        return {
            "d_model": cfg.MODEL.MESHEDMEORY.DECODER.DIM_MODEL,
            "num_layer": cfg.MODEL.MESHEDMEORY.DECODER.NUM_LAYER,
            "num_att_head": cfg.MODEL.MESHEDMEORY.DECODER.NUM_ATT_HEAD,
            "d_ff": cfg.MODEL.MESHEDMEORY.DECODER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.MESHEDMEORY.DECODER.DROPOUT,
            "padding_idx": -1, # default
            "enc_layer_num": cfg.MODEL.MESHEDMEORY.ENCODER.NUM_LAYER
        }

    @classmethod
    def add_config(cls, cfg):
        if not hasattr(cfg.MODEL, "MESHEDMEORY"):
            cfg.MODEL.MESHEDMEORY = CN()

        cfg.MODEL.MESHEDMEORY.DECODER = CN()
        cfg.MODEL.MESHEDMEORY.DECODER.DIM_MODEL = 512
        cfg.MODEL.MESHEDMEORY.DECODER.NUM_LAYER = 3
        cfg.MODEL.MESHEDMEORY.DECODER.DROPOUT = 0.1
        cfg.MODEL.MESHEDMEORY.DECODER.NUM_ATT_HEAD = 8
        cfg.MODEL.MESHEDMEORY.DECODER.DIM_FEEDFORWARD = 2048

    def forward(self, batched_inputs):
        ret = {}
        vfeats = batched_inputs[kfg.ATT_FEATS]
        vmasks = batched_inputs[kfg.ATT_MASKS]

        history_states = batched_inputs.get(kfg.HISTORY_STATES, None)

        g_tfeats_arr = []
        g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
        ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]
        ext_g_tmasks = (ext_g_tmasks == -10000.0) # FIXME
        if len(g_tfeats.size()) == 2:
            g_tfeats = g_tfeats.unsqueeze(1)

        if kfg.TIME_STEP in batched_inputs:
            time_step = batched_inputs[kfg.TIME_STEP]
            ext_g_tmasks = ext_g_tmasks[:,:, time_step:time_step+1, 0:time_step+1]
            if kfg.HISTORY_STATES not in batched_inputs:
                shape = list(g_tfeats.size())
                shape[1] = 0
                history_states = [g_tfeats.new(torch.Size(shape))] * self.num_layers
                batched_inputs[kfg.HISTORY_STATES] = history_states
        else:
            history_states = [None] * self.num_layers

        for i, layer_module in enumerate(self.layers):
            if history_states[i] is not None:
                history_states[i] = torch.cat([history_states[i], g_tfeats], dim=1)

            g_tfeats = layer_module(g_tfeats, vfeats, ext_g_tmasks, vmasks, history_states[i])
            g_tfeats_arr.append(g_tfeats)
        ret.update({ kfg.G_HIDDEN_STATES: g_tfeats_arr })

        return ret