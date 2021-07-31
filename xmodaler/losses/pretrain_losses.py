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
from .triplet import BatchTriplet

@LOSSES_REGISTRY.register()
class PretrainLosses(nn.Module):
    @configurable
    def __init__(self, margin, max_violation):
        super(PretrainLosses, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.triplet_loss = BatchTriplet(margin, max_violation)

    @classmethod
    def from_config(cls, cfg):
        return {
            "margin": cfg.LOSSES.MARGIN,
            "max_violation": cfg.LOSSES.MAX_VIOLATION
        }

    def select_logits_targets_by_mask(self, tensor, target, mask):
        tensor = tensor[mask, :]
        target = target[mask]
        return tensor, target

    def forward(self, batched_inputs):
        ret = {}

        if kfg.OUTPUT in batched_inputs:
            triplet_loss = self.triplet_loss(batched_inputs)
            triplet_loss['BatchTriplet Loss'] /= len(batched_inputs[kfg.IDS])
            ret.update(triplet_loss)

        if kfg.V_LOGITS in batched_inputs:
            v_logits = batched_inputs[kfg.V_LOGITS]
            v_logits = v_logits[:, 1:, :].reshape(-1, v_logits.size(-1))
            v_targets = batched_inputs[kfg.V_TARGET]
            v_targets = v_targets.view(-1, v_targets.size(-1))
            v_targets_labels = batched_inputs[kfg.V_TARGET_LABELS].view(-1)
            v_logits, v_targets = self.select_logits_targets_by_mask(v_logits, v_targets, v_targets_labels > 0)
            if v_targets.size(0) > 0:
                v_loss = self.kl_loss(F.log_softmax(v_logits, dim=-1), v_targets)
                v_loss = torch.sum(v_loss) / v_loss.size(0)
                ret.update({ "Masked Object Classification": v_loss })

        if kfg.U_LOGITS in batched_inputs:
            u_tlogits = batched_inputs[kfg.U_LOGITS]
            u_tlogits = u_tlogits.view(-1, u_tlogits.size(-1))
            u_targets = batched_inputs[kfg.U_TARGET_IDS].view(-1)
            u_tlogits, u_targets = self.select_logits_targets_by_mask(u_tlogits, u_targets, u_targets >= 0)
            if u_targets.size(0) > 0:
                u_loss = self.xe_loss(u_tlogits, u_targets)
                ret.update({ "Masked Language Modeling": u_loss })
                
        if kfg.G_LOGITS in batched_inputs:
            g_tlogits = batched_inputs[kfg.G_LOGITS]
            g_tlogits = g_tlogits.view(-1, g_tlogits.size(-1))
            g_targets = batched_inputs[kfg.G_TARGET_IDS].view(-1)
            g_tlogits, g_targets = self.select_logits_targets_by_mask(g_tlogits, g_targets, g_targets >= 0)
            if g_targets.size(0) > 0:
                g_loss = self.xe_loss(g_tlogits, g_targets)
                ret.update({ "Masked Sentence Generation": g_loss })

        return ret