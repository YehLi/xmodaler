# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

LOSSES_REGISTRY = Registry("LOSSES")
LOSSES_REGISTRY.__doc__ = """
Registry for losses
"""

def build_losses(cfg):
    losses = []
    for name in cfg.LOSSES.NAMES:
        loss = LOSSES_REGISTRY.get(name)(cfg)
        losses.append(loss)
    return losses

def build_rl_losses(cfg):
    losses = []
    loss = LOSSES_REGISTRY.get("RewardCriterion")(cfg)
    losses.append(loss)
    for name in cfg.LOSSES.NAMES:
        if name not in {'CrossEntropy', 'LabelSmoothing'}:
            loss = LOSSES_REGISTRY.get(name)(cfg)
            losses.append(loss)
    return losses

def add_loss_config(cfg, tmp_cfg):
    for name in tmp_cfg.LOSSES.NAMES:
        LOSSES_REGISTRY.get(name).add_config(cfg)