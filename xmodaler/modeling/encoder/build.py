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
    encoder = ENCODER_REGISTRY.get(cfg.MODEL.ENCODER)(cfg)
    return encoder

def add_encoder_config(cfg, tmp_cfg):
    ENCODER_REGISTRY.get(tmp_cfg.MODEL.ENCODER).add_config(cfg)