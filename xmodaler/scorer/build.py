# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

SCORER_REGISTRY = Registry("SCORER")
SCORER_REGISTRY.__doc__ = """
Registry for scorer
"""

def build_scorer(cfg):
    scorer = SCORER_REGISTRY.get(cfg.SCORER.NAME)(cfg)
    return scorer