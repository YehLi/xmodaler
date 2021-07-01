import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class PretrainLosses(nn.Module):
    @configurable
    def __init__(self):
        super(PretrainLosses, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.kl_loss = nn.KLDivLoss(reduction="none")

    @classmethod
    def from_config(cls, cfg):
        return {}

    def forward(self, batched_inputs):
        ret = {}
        if kfg.V_LOGITS in batched_inputs:
            v_logits = batched_inputs[kfg.V_LOGITS]
            v_targets = batched_inputs[kfg.V_TARGET]
            v_targets_labels = batched_inputs[kfg.V_TARGET_LABELS]

        if kfg.U_LOGITS in batched_inputs:
            u_tlogits = batched_inputs[kfg.U_LOGITS]
            u_targets = batched_inputs[kfg.U_TARGET_IDS]

        if kfg.G_LOGITS in batched_inputs:
            g_tlogits = batched_inputs[kfg.G_LOGITS]
            g_targets = batched_inputs[kfg.G_TARGET_IDS]

        return ret