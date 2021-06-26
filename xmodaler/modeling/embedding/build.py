import torch
from xmodaler.utils.registry import Registry

EMBEDDING_REGISTRY = Registry("EMBEDDING")
EMBEDDING_REGISTRY.__doc__ = """
Registry for embedding
"""

def build_embeddings(cfg, name):
    embeddings = EMBEDDING_REGISTRY.get(name)(cfg)
    return embeddings