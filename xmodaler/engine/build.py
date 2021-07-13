# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

ENGINE_REGISTRY = Registry("ENGINE")
ENGINE_REGISTRY.__doc__ = """
Registry for engine
"""

def build_engine(cfg):
    engine = ENGINE_REGISTRY.get(cfg.ENGINE.NAME)(cfg)
    return engine