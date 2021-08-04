# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from xmodaler.config import configurable
from .build import SOLVER_REGISTRY

@SOLVER_REGISTRY.register()
class Adam(torch.optim.Adam):
    @configurable
    def __init__(
        self, 
        *,
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8,
        weight_decay=0, 
        amsgrad=False
    ):
        super(Adam, self).__init__(
            params, 
            lr, 
            betas, 
            eps,
            weight_decay, 
            amsgrad
        )

    @classmethod
    def from_config(cls, cfg, params):
        return {
            "params": params,
            "lr": cfg.SOLVER.BASE_LR, 
            "betas": cfg.SOLVER.BETAS, 
            "eps": cfg.SOLVER.EPS,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY, 
            "amsgrad": cfg.SOLVER.AMSGRAD
        }
