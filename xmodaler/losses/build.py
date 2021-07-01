import torch

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