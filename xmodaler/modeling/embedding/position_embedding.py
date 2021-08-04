# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn

from xmodaler.utils.registry import Registry

POSITION_ENC_REGISTRY = Registry("POSITION_ENC")
POSITION_ENC_REGISTRY.__doc__ = """
Registry for positional encoding
"""

__all__ = ["SinusoidEncoding", "NNEmbeddingEncoding"]

def build_position_encoding(cfg, dim, max_len):
    name = cfg.MODEL.TOKEN_EMBED.POSITION
    return POSITION_ENC_REGISTRY.get(name)(dim, max_len)

@POSITION_ENC_REGISTRY.register()
class SinusoidEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(SinusoidEncoding, self).__init__()   
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() *
                             -(math.log(max_len * 2.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if isinstance(x, int):
            return self.pe[:, x]
        else:
            x_size = x.size(1)
            return self.pe[:, :x_size]

@POSITION_ENC_REGISTRY.register()
class NNEmbeddingEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(NNEmbeddingEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)

    def forward(self, x):
        if isinstance(x, int):
            position_embeddings = self.position_embeddings(torch.tensor([x], dtype=torch.long).cuda())
        else:
            x_size = x.size(1)
            position_ids = torch.arange(x_size, dtype=torch.long, device=x.device)
            position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings