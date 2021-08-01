# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
import random
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .decoder import Decoder
from ..layers.bert import BertUnderstandingLayer, BertGenerationLayer
from .build import DECODER_REGISTRY

__all__ = ["DecoupleBertDecoder"]

@DECODER_REGISTRY.register()
class DecoupleBertDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
       num_understanding_layers: int,
       num_generation_layers: int,
       u_layer_drop: float,
       g_layer_drop: float,
       bert_understanding_layers,
       bert_generation_layers
    ):
        super(DecoupleBertDecoder, self).__init__()
        self.num_understanding_layers = num_understanding_layers
        self.num_generation_layers = num_generation_layers
        self.u_layer_drop = u_layer_drop
        self.g_layer_drop = g_layer_drop
        if self.num_understanding_layers > 0:
            self.u_layers = bert_understanding_layers
        if self.num_generation_layers > 0:
            self.g_layers = bert_generation_layers
        
    @classmethod
    def from_config(cls, cfg):
        bert_understanding_layers = nn.ModuleList(
            [BertUnderstandingLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_UNDERSTANDING_LAYERS)]
        )
        bert_generation_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_GENERATION_LAYERS)]
        )
        return {
            "num_understanding_layers": cfg.MODEL.BERT.NUM_UNDERSTANDING_LAYERS,
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
            "u_layer_drop": cfg.MODEL.BERT.U_LAYER_DROP,
            "g_layer_drop": cfg.MODEL.BERT.G_LAYER_DROP,
            "bert_understanding_layers": bert_understanding_layers,
            "bert_generation_layers": bert_generation_layers,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ret = {}
        vfeats = batched_inputs[kfg.ATT_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]

        if isinstance(vfeats, list):
            vfeats = vfeats[-1]

        if (kfg.G_TOKEN_EMBED not in batched_inputs) and (self.num_generation_layers > 0):
            batched_inputs[kfg.G_TOKEN_EMBED] = batched_inputs[kfg.U_TOKEN_EMBED]
        if (kfg.U_TOKEN_EMBED not in batched_inputs) and (self.num_understanding_layers > 0):
            batched_inputs[kfg.U_TOKEN_EMBED] = batched_inputs[kfg.G_TOKEN_EMBED]

        if kfg.U_TOKEN_EMBED in batched_inputs:
            u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
            ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]
            if isinstance(u_tfeats, list):
                u_tfeats = u_tfeats[-1]

            vfeats_arr = []
            u_tfeats_arr = []
            for layer_module in self.u_layers:
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.u_layer_drop):
                    vfeats_arr.append(vfeats)
                    u_tfeats_arr.append(u_tfeats)
                else:
                    vfeats, u_tfeats = layer_module(vfeats, ext_vmasks, u_tfeats, ext_u_tmasks)
                    vfeats_arr.append(vfeats)
                    u_tfeats_arr.append(u_tfeats)
            ret.update({ kfg.U_HIDDEN_STATES: u_tfeats_arr, kfg.ATT_FEATS: vfeats_arr })

        if kfg.G_TOKEN_EMBED in batched_inputs:
            g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
            ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]
            if isinstance(g_tfeats, list):
                g_tfeats = g_tfeats[-1]
            if len(g_tfeats.size()) == 2:
                g_tfeats = g_tfeats.unsqueeze(1)

            history_states = batched_inputs.get(kfg.HISTORY_STATES, None)
            if kfg.TIME_STEP in batched_inputs:
                time_step = batched_inputs[kfg.TIME_STEP]
                ext_g_tmasks = ext_g_tmasks[:,:, time_step:time_step+1, 0:time_step+1]
                if kfg.HISTORY_STATES not in batched_inputs:
                    shape = list(g_tfeats.size())
                    shape[1] = 0
                    history_states = [g_tfeats.new(torch.Size(shape))] * self.num_generation_layers
                    batched_inputs[kfg.HISTORY_STATES] = history_states
            else:
                history_states = [None] * self.num_generation_layers

            g_tfeats_arr = []
            for i, layer_module in enumerate(self.g_layers):
                if history_states[i] is not None:
                    history_states[i] = torch.cat([history_states[i], g_tfeats], dim=1)

                g_tfeats = layer_module(g_tfeats, vfeats, ext_g_tmasks, ext_vmasks, history_states[i])
                g_tfeats_arr.append(g_tfeats)
            ret.update({ kfg.G_HIDDEN_STATES: g_tfeats_arr })

        return ret