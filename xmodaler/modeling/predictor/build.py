# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

PREDICTOR_REGISTRY = Registry("PREDICTOR")
PREDICTOR_REGISTRY.__doc__ = """
Registry for PREDICTOR
"""

def build_predictor(cfg):
    predictor = PREDICTOR_REGISTRY.get(cfg.MODEL.PREDICTOR)(cfg) if len(cfg.MODEL.PREDICTOR) > 0 else None
    return predictor

def build_v_predictor(cfg):
    predictor = PREDICTOR_REGISTRY.get(cfg.MODEL.V_PREDICTOR)(cfg) if len(cfg.MODEL.V_PREDICTOR) > 0 else None
    return predictor  

def build_predictor_with_name(cfg, name):
    predictor = PREDICTOR_REGISTRY.get(name)(cfg) if len(name) > 0 else None
    return predictor

def add_predictor_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.PREDICTOR) > 0:
        PREDICTOR_REGISTRY.get(tmp_cfg.MODEL.PREDICTOR).add_config(cfg)