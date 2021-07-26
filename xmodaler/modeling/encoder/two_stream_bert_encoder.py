# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import random
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer
from .build import ENCODER_REGISTRY

__all__ = ["TwoStreamBertEncoder"]

@ENCODER_REGISTRY.register()
class TwoStreamBertEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        v_num_hidden_layers: int,
        layer_drop: float,
        v_layer_drop: float,
        bert_layers,
        v_bert_layers
    ):
        super(TwoStreamBertEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.v_num_hidden_layers = v_num_hidden_layers
        self.layer_drop = layer_drop
        self.v_layer_drop = v_layer_drop

        self.layers = bert_layers
        self.v_layers = v_bert_layers
        
    @classmethod
    def from_config(cls, cfg):
        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )
        v_bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.V_NUM_HIDDEN_LAYERS)]
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "v_num_hidden_layers": cfg.MODEL.BERT.V_NUM_HIDDEN_LAYERS,
            "layer_drop": cfg.MODEL.BERT.LAYER_DROP,
            "v_layer_drop": cfg.MODEL.BERT.V_LAYER_DROP,
            "bert_layers": bert_layers,
            "v_bert_layers": v_bert_layers,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):  # 'v', 't'
        ret = {}
        if mode == None or mode == 'v':
            vfeats = batched_inputs[kfg.ATT_FEATS]
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]

            vfeats_arr = []
            for layer_module in self.v_layers:
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.v_layer_drop):
                    vfeats_arr.append(vfeats)
                else:
                    vfeats, _ = layer_module(vfeats, ext_vmasks)
                    vfeats_arr.append(vfeats)
            ret.update({ kfg.ATT_FEATS: vfeats_arr })

        elif mode == 't':
            if kfg.U_TOKEN_EMBED in batched_inputs:
                u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
                ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]

                u_tfeats_arr = []
                for layer_module in self.layers:
                    dropout_probability = random.uniform(0, 1)
                    if self.training and (dropout_probability < self.layer_drop):
                        u_tfeats_arr.append(u_tfeats)
                    else:
                        u_tfeats, _ = layer_module(u_tfeats, ext_u_tmasks)
                        u_tfeats_arr.append(u_tfeats)
                ret.update({ kfg.U_TOKEN_EMBED: u_tfeats_arr })

            if kfg.G_TOKEN_EMBED in batched_inputs:
                g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
                ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]

                g_tfeats_arr = []
                for layer_module in self.layers:
                    dropout_probability = random.uniform(0, 1)
                    if self.training and (dropout_probability < self.layer_drop):
                        g_tfeats_arr.append(g_tfeats)
                    else:
                        g_tfeats, _ = layer_module(g_tfeats, ext_g_tmasks)
                        g_tfeats_arr.append(g_tfeats)
                ret.update({ kfg.G_TOKEN_EMBED: g_tfeats_arr })

        return ret