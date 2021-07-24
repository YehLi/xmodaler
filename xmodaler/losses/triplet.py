# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class BatchTriplet(nn.Module):
    @configurable
    def __init__(self, margin, max_violation):
        super(BatchTriplet, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    @classmethod
    def from_config(cls, cfg):
        return {
            "margin": cfg.LOSSES.MARGIN,
            "max_violation": cfg.LOSSES.MAX_VIOLATION
        }

    def forward(self, outputs_dict):
        scores = outputs_dict[kfg.OUTPUT]
        ids = np.array(outputs_dict[kfg.IDS])
        labels = np.expand_dims(ids, axis=1) == np.expand_dims(ids, axis=0)
        labels = torch.from_numpy(labels).cuda()

        # compute image-sentence score matrix
        batch_size = scores.size(0)
        diagonal = scores.diag().view(batch_size, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s = cost_s.masked_fill_(labels, 0)
        cost_im = cost_im.masked_fill_(labels, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            triplet_num = len(cost_s) + len(cost_im)
            loss = cost_s.sum() + cost_im.sum()
        else:
            cost_s = torch.masked_select(cost_s, cost_s > 0)
            cost_im = torch.masked_select(cost_im, cost_im > 0)
            triplet_num = len(cost_s) + len(cost_im)
            loss = cost_s.mean() + cost_im.mean()

        return { 'BatchTriplet Loss': loss, "Triplet_num": triplet_num }