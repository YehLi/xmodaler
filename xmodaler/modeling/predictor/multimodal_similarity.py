# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .build import PREDICTOR_REGISTRY
from ..layers.attention_pooler import AttentionPooler
from .multimodal_predictor import SingleStreamMultiModalPredictor

__all__ = ["MultiModalSimilarity", "SingleStreamMultiModalSimilarity"]

@PREDICTOR_REGISTRY.register()
class MultiModalSimilarity(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        pooler_input_size: int,
        pooler_output_size: int,
        pooler_bn: bool,
        pooler_dropout: float,
        num_hidden_layers: int,
        v_num_hidden_layers: int,
    ):
        super(MultiModalSimilarity, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.v_num_hidden_layers = v_num_hidden_layers
        
        self.t_pooler = AttentionPooler(
            hidden_size = pooler_input_size, 
            output_size = pooler_output_size,
            dropout = pooler_dropout,
            use_bn = pooler_bn
        )

        self.v_pooler = nn.ModuleList(
            [
                AttentionPooler(
                    hidden_size = pooler_input_size, 
                    output_size = pooler_output_size,
                    dropout = pooler_dropout,
                    use_bn = pooler_bn
                ) for _ in range(self.v_num_hidden_layers)
            ]
        )
        
    @classmethod
    def from_config(cls, cfg):

        return {
            "pooler_input_size": cfg.MODEL.MM_PREDICTOR.POOLER_INPUT_SIZE,
            "pooler_output_size": cfg.MODEL.MM_PREDICTOR.POOLER_OUTPUT_SIZE,
            "pooler_bn": cfg.MODEL.MM_PREDICTOR.POOLER_BN,
            "pooler_dropout": cfg.MODEL.MM_PREDICTOR.POOLER_DROPOUT,
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "v_num_hidden_layers": cfg.MODEL.BERT.V_NUM_HIDDEN_LAYERS,
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.MM_PREDICTOR = CN()
        cfg.MODEL.MM_PREDICTOR.POOLER_INPUT_SIZE = 768
        cfg.MODEL.MM_PREDICTOR.POOLER_OUTPUT_SIZE = 768
        cfg.MODEL.MM_PREDICTOR.POOLER_BN = False
        cfg.MODEL.MM_PREDICTOR.POOLER_DROPOUT = 0.1

    def forward(self, batched_inputs):
        u_tfeats = batched_inputs[kfg.U_TOKEN_EMBED]
        ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]
        vfeats_arr = batched_inputs[kfg.ATT_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]

        if isinstance(u_tfeats, list):
            u_tfeats = u_tfeats[-1]

        vfeats = 0
        for i in range(self.v_num_hidden_layers):
            vfeats = vfeats + self.v_pooler[i](vfeats_arr[i], ext_vmasks)
        vfeats = vfeats / np.sqrt(self.v_num_hidden_layers)
        vfeats = F.normalize(vfeats, p=2, dim=1)
        u_tfeats = self.t_pooler(u_tfeats, ext_u_tmasks)
        u_tfeats = F.normalize(u_tfeats, p=2, dim=1)

        if self.training:
            similarity = (u_tfeats.unsqueeze(1) * vfeats.unsqueeze(0)).sum(dim=-1)
            return { kfg.OUTPUT: similarity }
        else:
            return { kfg.OUTPUT: [vfeats, u_tfeats] }


@PREDICTOR_REGISTRY.register()
class SingleStreamMultiModalSimilarity(SingleStreamMultiModalPredictor):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        labels_num: int,
        pooler
    ):
        super(SingleStreamMultiModalSimilarity, self).__init__(
            hidden_size=hidden_size,
            labels_num=labels_num,
            pooler=pooler
        )

    def test_forward(self, u_logits):
        # for Single stream similarity
        return { kfg.OUTPUT: u_logits }