# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jingwen Chen
@contact: chenjingwen.sysu@gmail.com
"""
import torch
from torch import nn
import math 
from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers import TemporalDeformableLayer
from .build import ENCODER_REGISTRY

__all__ = ["TDConvEDEncoder"]

@ENCODER_REGISTRY.register()
class TDConvEDEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        hidden_size: int,
        kernel_sizes: list, # list of int
        padding_mode: str, # 'border'
        offset_act: str, # 'tanh'
        min_idx: int,
        max_idx: int,
        clamp_idx: bool,
        dropout: float,
        use_norm: bool
    ):
        super(TDConvEDEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.padding_mode = padding_mode
        self.offset_act = offset_act
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.clamp_idx = clamp_idx

        self.layers = nn.ModuleList(
                        [
                            TemporalDeformableLayer(
                                hidden_size,
                                hidden_size,
                                kernel_size,
                                1,
                                self.padding_mode, # 'border'
                                self.offset_act,
                                self.min_idx,
                                self.max_idx,
                                self.clamp_idx,
                                dropout,
                                use_norm
                            ) for kernel_size in self.kernel_sizes
                        ]
                    )
        

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_hidden_layers": cfg.MODEL.TDCONVED.ENCODER.NUM_HIDDEN_LAYERS,
            "hidden_size": cfg.MODEL.TDCONVED.ENCODER.HIDDEN_SIZE,
            "kernel_sizes": cfg.MODEL.TDCONVED.ENCODER.KERNEL_SIZES, # list of int
            "padding_mode": cfg.MODEL.TDCONVED.ENCODER.PADDING_MODE, # 'border'
            "offset_act": cfg.MODEL.TDCONVED.ENCODER.OFFSET_ACT, # 'tanh'
            "min_idx": cfg.MODEL.TDCONVED.ENCODER.OFFSET_MIN,
            "max_idx": cfg.MODEL.TDCONVED.ENCODER.OFFSET_MAX,
            "clamp_idx": cfg.MODEL.TDCONVED.ENCODER.CLAMP_OFFSET,
            "dropout": cfg.MODEL.TDCONVED.ENCODER.DROPOUT,
            "use_norm": cfg.MODEL.TDCONVED.ENCODER.USE_NORM
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.TDCONVED = CN()

        cfg.MODEL.TDCONVED.ENCODER = CN()
        cfg.MODEL.TDCONVED.ENCODER.NUM_HIDDEN_LAYERS = 2
        cfg.MODEL.TDCONVED.ENCODER.HIDDEN_SIZE = 512
        cfg.MODEL.TDCONVED.ENCODER.KERNEL_SIZES = [3, 3]
        cfg.MODEL.TDCONVED.ENCODER.PADDING_MODE = 'border'
        cfg.MODEL.TDCONVED.ENCODER.OFFSET_ACT = 'tanh'
        cfg.MODEL.TDCONVED.ENCODER.OFFSET_MIN = -1.0
        cfg.MODEL.TDCONVED.ENCODER.OFFSET_MAX = 1.0
        cfg.MODEL.TDCONVED.ENCODER.CLAMP_OFFSET = True
        cfg.MODEL.TDCONVED.ENCODER.DROPOUT = 0.5
        cfg.MODEL.TDCONVED.ENCODER.USE_NORM = True 

    def forward(self, batched_inputs, mode=None):
        if mode == 't':
            return {}
        
        vfeats = batched_inputs[kfg.ATT_FEATS]
        masks = batched_inputs[kfg.ATT_MASKS]

        layer_input = vfeats
        layer_outputs = []
        for layer_module in self.layers:
            layer_output = layer_module(layer_input)
            layer_output = (layer_output + layer_input) * math.sqrt(0.5)
            layer_outputs.append(layer_output)
            layer_input = layer_output

        return {kfg.ATT_FEATS: layer_output}
