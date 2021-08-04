# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from xmodaler.config import configurable
from .build import SOLVER_REGISTRY

@SOLVER_REGISTRY.register()
class RMSprop(torch.optim.RMSprop):
    @configurable
    def __init__(
        self, 
        *,
        params, 
        lr=1e-2, 
        alpha=0.99,
        eps=1e-8,
        weight_decay=0, 
        momentum=0,
        centered=False
    ):
        super(RMSprop, self).__init__(
            params, 
            lr, 
            alpha, 
            eps,
            weight_decay, 
            momentum,
            centered
        )

    @classmethod
    def from_config(cls, cfg, params):
        return {
            "params": params,
            "lr": cfg.SOLVER.BASE_LR, 
            "alpha": cfg.SOLVER.ALPHA, 
            "eps": cfg.SOLVER.EPS,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY, 
            "momentum": cfg.SOLVER.MOMENTUM,
            "centered": cfg.SOLVER.CENTERED
        }
