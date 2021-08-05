# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..layers.bert import BertLayer
from .build import ENCODER_REGISTRY

__all__ = ["SingleStreamBertEncoder"]

@ENCODER_REGISTRY.register()
class SingleStreamBertEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        bert_layers,
    ):
        super(SingleStreamBertEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers
        
    @classmethod
    def from_config(cls, cfg):
        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "bert_layers": bert_layers
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):  # 'v', 't', 'vt'
        ret = {}

        vfeats = batched_inputs[kfg.ATT_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]

        u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
        ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]

        lang_token_num = u_tfeats.size(1)
        lv_feats = torch.cat([u_tfeats, vfeats], dim=1)
        lv_attention_mask = torch.cat([ext_u_tmasks, ext_vmasks], dim=-1)

        for layer_module in self.layers:
            lv_feats, _ = layer_module(lv_feats, lv_attention_mask)

        lang_feats = lv_feats[:, :lang_token_num]
        v_feats = lv_feats[:, lang_token_num:]

        ret.update({
            kfg.ATT_FEATS: v_feats,
            kfg.U_HIDDEN_STATES: lang_feats
        })
        
        return ret