# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from xmodaler.config import configurable
from torch.nn.utils import clip_grad_norm_
from .build import SOLVER_REGISTRY

@SOLVER_REGISTRY.register()
class BertAdam(torch.optim.Optimizer):
    @configurable
    def __init__(
        self, 
        *,
        params, 
        lr=1e-3, 
        b1=0.9,
        b2=0.999, 
        eps=1e-6,
        weight_decay=0.01, 
        max_grad_norm=1.0
    ):
        
        defaults = dict(lr=lr, b1=b1, b2=b2, e=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)
        

    @classmethod
    def from_config(cls, cfg, params):
        return {
            "params": params,
            "lr": cfg.SOLVER.BASE_LR, 
            "b1": cfg.SOLVER.BETAS[0],
            "b2": cfg.SOLVER.BETAS[1],
            "eps": 1e-6,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY, 
            "max_grad_norm": 1.0
        }

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

        return loss