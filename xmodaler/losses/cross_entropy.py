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
class CrossEntropy(nn.Module):
    @configurable
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def from_config(cls, cfg):
        return {}

    def forward(self, outputs_dict):
        ret  = {}
        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            targets = outputs_dict[kfg.G_TARGET_IDS]

            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1).long()
            loss = self.criterion(logits, targets)
            ret.update({ 'CrossEntropy Loss(G)': loss })

        if kfg.U_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.U_LOGITS]
            targets = outputs_dict[kfg.U_TARGET_IDS]

            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1).long()
            loss = self.criterion(logits, targets)
            ret.update({'CrossEntropy Loss(U)': loss})
            
        return ret

