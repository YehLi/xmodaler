import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .build import ENCODER_REGISTRY

__all__ = ["UpDownEncoder"]

@ENCODER_REGISTRY.register()
class UpDownEncoder(nn.Module):
    @configurable
    def __init__(self):
        super(UpDownEncoder, self).__init__()
        
    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            att_feats = batched_inputs[kfg.ATT_FEATS]
            att_masks = batched_inputs[kfg.ATT_MASKS]
            if att_masks is None:
                global_feats = torch.mean(att_feats, 1)
            else:
                att_feats_masks = att_feats * att_masks.unsqueeze(-1)
                att_masks_sum = att_masks.sum(-1)
                global_feats = att_feats_masks.sum(1) / att_masks_sum.unsqueeze(-1)
            ret.update({ kfg.GLOBAL_FEATS: global_feats })
        
        return ret