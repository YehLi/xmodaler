# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class LabelSmoothing(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        label_smoothing
    ):
        super(LabelSmoothing, self).__init__()
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - self.label_smoothing
        self.criterion = nn.KLDivLoss(reduction='none')

    @classmethod
    def from_config(cls, cfg):
        return {
            "label_smoothing": cfg.LOSSES.LABELSMOOTHING
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def Forward(self, logits, targets):
        logP = F.log_softmax(logits.view(-1, logits.shape[-1]), dim=-1) 
        targets = targets.view(-1)
        mask = targets >= 0

        assign_seq = targets  #.type(torch.cuda.LongTensor)
        assign_seq[assign_seq < 0] = 0

        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss = self.criterion(logP, true_dist).sum(1)
        loss = torch.masked_select(loss, mask).mean()
        return loss

    def forward(self, outputs_dict):
        ret  = {}
        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            targets = outputs_dict[kfg.G_TARGET_IDS]
            loss = self.Forward(logits, targets)
            ret.update({ 'LabelSmoothing(G) loss': loss })

        if kfg.U_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.U_LOGITS]
            targets = outputs_dict[kfg.U_TARGET_IDS]
            loss = self.Forward(logits, targets)
            ret.update({ 'LabelSmoothing(U) loss': loss })
        return ret