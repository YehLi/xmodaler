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
from .build import PREDICTOR_REGISTRY
from ..layers.attention_pooler import AttentionPooler

__all__ = ["MultiModalPredictor"]

@PREDICTOR_REGISTRY.register()
class MultiModalPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        labels_num: int,
        pooler_input_size: int,
        pooler_output_size: int,
        pooler_bn: bool,
        pooler_dropout: float,
        num_understanding_layers: int,
        num_generation_layers: int,
    ):
        super(MultiModalPredictor, self).__init__()
        self.num_understanding_layers = num_understanding_layers
        self.num_generation_layers = num_generation_layers
        
        if self.num_understanding_layers > 0:
            self.u_pooler = AttentionPooler(
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
                    )
                    for _ in range(self.num_understanding_layers)
                ]
            )
            self.u_logits = nn.Sequential(
                nn.LayerNorm(pooler_output_size),
                nn.Linear(pooler_output_size, labels_num)
            )
        
        if self.num_generation_layers > 0:
            self.g_pooler = AttentionPooler(
                hidden_size = pooler_input_size, 
                output_size = pooler_output_size,
                dropout = pooler_dropout,
                use_bn = pooler_bn
            )
            self.g_logits = nn.Sequential(
                nn.LayerNorm(pooler_output_size),
                nn.Linear(pooler_output_size, labels_num)
            )

    @classmethod
    def from_config(cls, cfg):
        return {
            "labels_num": cfg.MODEL.MM_PREDICTOR.LABELS_NUM,
            "pooler_input_size": cfg.MODEL.MM_PREDICTOR.POOLER_INPUT_SIZE,
            "pooler_output_size": cfg.MODEL.MM_PREDICTOR.POOLER_OUTPUT_SIZE,
            "pooler_bn": cfg.MODEL.MM_PREDICTOR.POOLER_BN,
            "pooler_dropout": cfg.MODEL.MM_PREDICTOR.POOLER_DROPOUT,
            "num_understanding_layers": cfg.MODEL.BERT.NUM_UNDERSTANDING_LAYERS,
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.MM_PREDICTOR = CN()
        cfg.MODEL.MM_PREDICTOR.LABELS_NUM = 3129

        cfg.MODEL.MM_PREDICTOR.POOLER_INPUT_SIZE = 768
        cfg.MODEL.MM_PREDICTOR.POOLER_OUTPUT_SIZE = 768
        cfg.MODEL.MM_PREDICTOR.POOLER_BN = False
        cfg.MODEL.MM_PREDICTOR.POOLER_DROPOUT = 0.1

    def forward(self, batched_inputs):
        outputs = 0
        ret = {}
        if kfg.U_HIDDEN_STATES in batched_inputs:
            u_tfeats = batched_inputs[kfg.U_HIDDEN_STATES]
            ext_u_tmasks = batched_inputs[kfg.EXT_U_TOKENS_MASKS]
            vfeats_arr = batched_inputs[kfg.ATT_FEATS]
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]

            if isinstance(u_tfeats, list):
                u_tfeats = u_tfeats[-1]

            vfeats = 0
            for i in range(self.num_understanding_layers):
                vfeats = vfeats + self.v_pooler[i](vfeats_arr[i], ext_vmasks)
            u_tfeats = self.u_pooler(u_tfeats, ext_u_tmasks)
            pooled_output = vfeats * u_tfeats
            u_logits = self.u_logits(pooled_output)
            ret.update({ kfg.U_LOGITS: u_logits })

            if not self.training:
                outputs = outputs + torch.softmax(u_logits, dim=-1)

        if kfg.G_HIDDEN_STATES in batched_inputs:
            g_tfeats = batched_inputs[kfg.G_HIDDEN_STATES]
            ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]

            if isinstance(g_tfeats, list):
                g_tfeats = g_tfeats[-1]
            g_tfeats = self.g_pooler(g_tfeats, ext_g_tmasks)
            g_logits = self.g_logits(g_tfeats)
            ret.update({ kfg.G_LOGITS: g_logits })
            
            if not self.training:
                outputs = outputs + torch.softmax(g_logits, dim=-1)
                outputs = torch.max(outputs, 1)[1].data.cpu().numpy()

        ret.update({ kfg.OUTPUT: outputs })
        return ret