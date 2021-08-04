# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoder
"""

def build_encoder(cfg):
    encoder = ENCODER_REGISTRY.get(cfg.MODEL.ENCODER)(cfg) if len(cfg.MODEL.ENCODER) > 0 else None
    return encoder

def add_encoder_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.ENCODER) > 0:
        ENCODER_REGISTRY.get(tmp_cfg.MODEL.ENCODER).add_config(cfg)