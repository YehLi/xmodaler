import torch
from xmodaler.config import configurable
from .build import LR_SCHEDULER_REGISTRY

@LR_SCHEDULER_REGISTRY.register()
class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    @configurable
    def __init__(
        self,
        *,
        optimizer,
        model_size,
        factor,
        warmup,
        last_epoch=-1,
    ):

        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer, last_epoch)

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "model_size": cfg.LR_SCHEDULER.MODEL_SIZE,
            "factor": cfg.LR_SCHEDULER.FACTOR,
            "warmup": cfg.LR_SCHEDULER.WARMUP, # iterations
            "last_epoch": -1
        }

    def get_lr(self):
        return [
            self.factor * \
            (self.model_size ** (-0.5) *
            min((self.last_epoch + 1) ** (-0.5), (self.last_epoch + 1) * self.warmup ** (-1.5)))
            for base_lr in self.base_lrs
        ]