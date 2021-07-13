# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from xmodaler.config import configurable
from .build import SOLVER_REGISTRY

@SOLVER_REGISTRY.register()
class SGD(torch.optim.SGD):
    @configurable
    def __init__(
        self, 
        *,
        params, 
        lr=0.1, 
        momentum=0, 
        dampening=0,
        weight_decay=0, 
        nesterov=False
    ):
        super(SGD, self).__init__(
            params, 
            lr, 
            momentum, 
            dampening,
            weight_decay, 
            nesterov
        )

    @classmethod
    def from_config(cls, cfg, params):
        return {
            "params": params,
            "lr": cfg.SOLVER.BASE_LR,
            "momentum": cfg.SOLVER.MOMENTUM,
            "dampening": cfg.SOLVER.DAMPENING,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "nesterov": cfg.SOLVER.NESTEROV
        }
