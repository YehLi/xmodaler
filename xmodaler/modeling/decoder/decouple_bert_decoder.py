import copy
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertUnderstandingLayer, BertGenerationLayer
from .build import DECODER_REGISTRY

__all__ = ["DecoupleBertDecoder"]

@DECODER_REGISTRY.register()
class DecoupleBertDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
       num_understanding_layers: int,
       num_generation_layers: int,
       bert_understanding_layer: BertUnderstandingLayer,
       bert_generation_layer: BertGenerationLayer
    ):
        super(DecoupleBertDecoder, self).__init__()
        self.num_understanding_layers = num_understanding_layers
        self.num_generation_layers = num_generation_layers
        if self.num_understanding_layers > 0:
            self.u_layers = nn.ModuleList(
                [copy.copy(bert_understanding_layer) for _ in range(self.num_understanding_layers)]
            )
        if self.num_generation_layers > 0:
            self.g_layers = nn.ModuleList(
                [copy.copy(bert_generation_layer) for _ in range(self.num_generation_layers)]
            )
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "num_understanding_layers": cfg.MODEL.BERT.NUM_UNDERSTANDING_LAYERS,
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
            "bert_understanding_layer": BertUnderstandingLayer(cfg),
            "bert_generation_layer": BertGenerationLayer(cfg),
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ret = {}
        vfeats = batched_inputs[kfg.ATT_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]

        if kfg.U_TOKEN_EMBED in batched_inputs:
            u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
            ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]
            for layer_module in self.u_layers:
                vfeats, u_tfeats = layer_module(vfeats, ext_vmasks, u_tfeats, ext_u_tmasks)
            ret.update({ kfg.U_HIDDEN_STATES: u_tfeats, kfg.ATT_FEATS: vfeats })

        if kfg.G_TOKEN_EMBED in batched_inputs:
            g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
            ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]
            for layer_module in self.g_layers:
                g_tfeats = layer_module(g_tfeats, vfeats, ext_g_tmasks, ext_vmasks)
            ret.update({ kfg.G_HIDDEN_STATES: g_tfeats })

        return ret