
"""
Paper: 'Asymmetric loss for multi-label classification'
       https://arxiv.org/abs/2009.14119
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.config import CfgNode as CN
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class SemComphderLoss(nn.Module):
    @configurable
    def __init__(self, filter_weight, reconstruct_weight, slot_size, num_classes):
        super(SemComphderLoss, self).__init__()
        weight = torch.ones((num_classes+1, )).cuda()
        weight[-1] = 30.0
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
        self.filter_weight = filter_weight
        self.reconstruct_weight = reconstruct_weight
        self.slot_size = slot_size

        self.gamma_neg = 5
        self.gamma_pos = 0
        self.clip = 0.05
        self.disable_torch_grad_focal_loss = True
        self.eps = 1e-8

    @classmethod
    def from_config(cls, cfg):
        return { 
            "filter_weight": cfg.MODEL.COSNET.FILTER_WEIGHT,
            "reconstruct_weight": cfg.MODEL.COSNET.RECONSTRUCT_WEIGHT,
            "slot_size": cfg.MODEL.COSNET.SLOT_SIZE,
            "num_classes": cfg.MODEL.COSNET.NUM_CLASSES
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret = {}
        logits = outputs_dict[kfg.SEMANTICS_PRED]
                
        semantics_logits = logits[:, self.slot_size:, :]
        semantics_logits = semantics_logits.reshape(-1, semantics_logits.shape[-1])
        semantics_labels = outputs_dict[kfg.SEMANTICS_LABELS].view(-1).long()
        filter_loss = self.criterion(semantics_logits, semantics_labels)

        memory_logits = logits[:, 0:self.slot_size, :]
        semantics_miss_labels = outputs_dict[kfg.SEMANTICS_MISS_LABELS]
        
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(memory_logits)
        x_sigmoid, _ = torch.max(x_sigmoid, dim=1)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = semantics_miss_labels * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - semantics_miss_labels) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                 torch.set_grad_enabled(False)
            pt0 = xs_pos * semantics_miss_labels
            pt1 = xs_neg * (1 - semantics_miss_labels)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * semantics_miss_labels + self.gamma_neg * (1 - semantics_miss_labels)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                 torch.set_grad_enabled(True)
            loss *= one_sided_w
        reconstruct_loss = -loss.sum(-1).mean()

        ret.update({
            "filter_loss": filter_loss * self.filter_weight,
            "reconstruct_loss": reconstruct_loss * self.reconstruct_weight
        })
        return ret