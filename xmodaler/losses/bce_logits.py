# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class BCEWithLogits(nn.Module):
    @configurable
    def __init__(self):
        super(BCEWithLogits, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        targets = outputs_dict[kfg.U_TARGET_IDS]

        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            loss = self.criterion(logits, targets) * targets.size(1)
            ret.update({ 'BCEWithLogits Loss(G)': loss })

        if kfg.U_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.U_LOGITS]
            loss = self.criterion(logits, targets) * targets.size(1)
            ret.update({ 'BCEWithLogits Loss(U)': loss })
            
        return ret