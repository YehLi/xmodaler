# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from xmodaler.config import configurable
from .build import SOLVER_REGISTRY

@SOLVER_REGISTRY.register()
class Adagrad(torch.optim.Adagrad):
    @configurable
    def __init__(
        self, 
        *,
        params, 
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10
    ):
        super(Adagrad, self).__init__(
            params, 
            lr, 
            lr_decay,
            weight_decay,
            initial_accumulator_value,
            eps
        )

    @classmethod
    def from_config(cls, cfg, params):
        return {
            "params": params,
            "lr": cfg.SOLVER.BASE_LR, 
            "lr_decay": cfg.SOLVER.LR_DECAY, 
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY, 
            "initial_accumulator_value": cfg.SOLVER.INITIAL_ACCUMULATOR_VALUE,
            "eps": cfg.SOLVER.EPS
        }
