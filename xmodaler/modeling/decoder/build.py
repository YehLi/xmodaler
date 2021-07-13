# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

DECODER_REGISTRY = Registry("DECODER")
DECODER_REGISTRY.__doc__ = """
Registry for decoder
"""

def build_decoder(cfg):
    decoder = DECODER_REGISTRY.get(cfg.MODEL.DECODER)(cfg)
    return decoder

def add_decoder_config(cfg, tmp_cfg):
    DECODER_REGISTRY.get(tmp_cfg.MODEL.DECODER).add_config(cfg)