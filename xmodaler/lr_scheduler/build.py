# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

LR_SCHEDULER_REGISTRY = Registry("LR_SCHEDULER")
LR_SCHEDULER_REGISTRY.__doc__ = """
Registry for lr scheduler
"""

def build_lr_scheduler(cfg, optimizer, data_size):
    lr_scheduler = LR_SCHEDULER_REGISTRY.get(cfg.LR_SCHEDULER.NAME)(cfg, optimizer, data_size)
    return lr_scheduler
