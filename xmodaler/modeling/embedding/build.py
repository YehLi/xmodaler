# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

EMBEDDING_REGISTRY = Registry("EMBEDDING")
EMBEDDING_REGISTRY.__doc__ = """
Registry for embedding
"""

def build_embeddings(cfg, name):
    embeddings = EMBEDDING_REGISTRY.get(name)(cfg)
    return embeddings