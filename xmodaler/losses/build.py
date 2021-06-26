import torch

from xmodaler.utils.registry import Registry

LOSSES_REGISTRY = Registry("LOSSES")
LOSSES_REGISTRY.__doc__ = """
Registry for losses
"""

def build_losses(cfg):
    losses = LOSSES_REGISTRY.get(cfg.LOSSES.NAME)(cfg)
    return losses