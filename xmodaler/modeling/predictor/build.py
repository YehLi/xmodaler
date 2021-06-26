import torch

from xmodaler.utils.registry import Registry

PREDICTOR_REGISTRY = Registry("PREDICTOR")
PREDICTOR_REGISTRY.__doc__ = """
Registry for PREDICTOR
"""

def build_predictor(cfg):
    predictor = PREDICTOR_REGISTRY.get(cfg.MODEL.PREDICTOR)(cfg)
    return predictor

def add_predictor_config(cfg, tmp_cfg):
    PREDICTOR_REGISTRY.get(tmp_cfg.MODEL.PREDICTOR).add_config(cfg)