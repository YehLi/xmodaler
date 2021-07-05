import copy
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
        bert_layer: BertLayer,
    ):
        super(TwoStreamBertEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.v_num_hidden_layers = v_num_hidden_layers

        self.layers = nn.ModuleList(
            [copy.copy(bert_layer) for _ in range(self.num_hidden_layers)]
        )
        self.v_layers = nn.ModuleList(
            [copy.copy(bert_layer) for _ in range(self.v_num_hidden_layers)]
        )
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "v_num_hidden_layers": cfg.MODEL.BERT.V_NUM_HIDDEN_LAYERS,
            "bert_layer": BertLayer(cfg),
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):  # 'v', 't'
        ret = {}
        if mode == None or mode == 'v':
            vfeats = batched_inputs[kfg.ATT_FEATS]
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
            for layer_module in self.v_layers:
                vfeats, _ = layer_module(vfeats, ext_vmasks)
            ret.update({ kfg.ATT_FEATS: vfeats })

        elif mode == 't':
            if kfg.U_TOKEN_EMBED in batched_inputs:
                u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
                ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]
                for layer_module in self.layers:
                    u_tfeats, _ = layer_module(u_tfeats, ext_u_tmasks)
                ret.update({ kfg.U_TOKEN_EMBED: u_tfeats })

            if kfg.G_TOKEN_EMBED in batched_inputs:
                g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
                ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]
                for layer_module in self.layers:
                    g_tfeats, _ = layer_module(g_tfeats, ext_g_tmasks)
                ret.update({ kfg.G_TOKEN_EMBED: g_tfeats })

        return ret