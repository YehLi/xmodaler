import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .build import ENCODER_REGISTRY

__all__ = ["Encoder"]

@ENCODER_REGISTRY.register()
class Encoder(nn.Module):
    @configurable
    def __init__(self):
        super(Encoder, self).__init__()
        
    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):
        return {}