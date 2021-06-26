import torch

from xmodaler.utils.registry import Registry

EVALUATION_REGISTRY = Registry("EVALUATION")
EVALUATION_REGISTRY.__doc__ = """
Registry for evaluation
"""

def build_evaluation(cfg, annfile):
    evaluation = EVALUATION_REGISTRY.get(cfg.INFERENCE.NAME)(cfg, annfile)
    return evaluation