import math
from bisect import bisect_right
import warnings
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from xmodaler.config import configurable
from .build import LR_SCHEDULER_REGISTRY

@LR_SCHEDULER_REGISTRY.register()
class WarmupConstant(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    @configurable
    def __init__(
        self, 
        *,
        optimizer, 
        warmup_steps, 
        last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstant, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "warmup_steps": cfg.LR_SCHEDULER.WARMUP * data_size,
            "last_epoch": -1
        }

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.

@LR_SCHEDULER_REGISTRY.register()
class WarmupLinear(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    @configurable
    def __init__(
        self, 
        *,
        optimizer, 
        min_lr, 
        warmup_steps,
        t_total, 
        last_epoch=-1):

        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_lr = min_lr
        super(WarmupLinear, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "min_lr": cfg.LR_SCHEDULER.MIN_LR,
            "warmup_steps": cfg.LR_SCHEDULER.WARMUP * data_size,
            "t_total": cfg.SOLVER.EPOCH * data_size, # total iterations
            "last_epoch": -1
        }

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(self.min_lr, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

@LR_SCHEDULER_REGISTRY.register()
class WarmupCosine(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    @configurable
    def __init__(
        self, 
        *,
        optimizer, 
        min_lr, 
        warmup_steps, 
        t_total, 
        cycles=.5, 
        last_epoch=-1):

        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        super(WarmupCosine, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "min_lr": cfg.LR_SCHEDULER.MIN_LR,
            "warmup_steps": cfg.LR_SCHEDULER.WARMUP * data_size,
            "t_total": cfg.SOLVER.EPOCH * data_size, # total iterations
            "cycles": .5, 
            "last_epoch": -1
        }

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(self.min_lr, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

@LR_SCHEDULER_REGISTRY.register()
class WarmupCosineWithHardRestarts(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """
    @configurable
    def __init__(
        self, 
        *,
        optimizer, 
        warmup_steps, 
        t_total, 
        cycles=1., 
        last_epoch=-1):

        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestarts, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "warmup_steps": cfg.LR_SCHEDULER.WARMUP * data_size,
            "t_total": cfg.SOLVER.EPOCH * data_size, # total iterations
            "cycles": 1., 
            "last_epoch": -1
        }

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))

@LR_SCHEDULER_REGISTRY.register()
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    @configurable
    def __init__(
        self,
        *,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        steps = [step * data_size for step in cfg.LR_SCHEDULER.STEPS]
        return {
            "optimizer": optimizer,
            "milestones": steps,
            "gamma": cfg.LR_SCHEDULER.GAMMA,
            "warmup_factor": cfg.LR_SCHEDULER.WARMUP_FACTOR,
            "warmup_iters": cfg.LR_SCHEDULER.WARMUP * data_size,
            "warmup_method": cfg.LR_SCHEDULER.WARMUP_METHOD,
            "last_epoch": -1
        }

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]