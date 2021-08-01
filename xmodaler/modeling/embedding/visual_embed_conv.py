# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["TDConvEDVisualBaseEmbedding"]

@EMBEDDING_REGISTRY.register()
class TDConvEDVisualBaseEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        in_dim: int,
        out_dim: int,
        **kwargs
    ):
        super(TDConvEDVisualBaseEmbedding, self).__init__()

        use_weightnorm = kwargs.pop("embeddings_weightnorm", False)
        if use_weightnorm:
            self.embeddings = weight_norm(nn.Linear(in_dim, out_dim))
        else:
            self.embeddings = nn.Linear(in_dim, out_dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "in_dim": cfg.MODEL.VISUAL_EMBED.IN_DIM,
            "out_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM
        }

        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.VISUAL_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.VISUAL_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.VISUAL_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.VISUAL_EMBED.USE_NORM:
            # embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            # kwargs['embeddings_norm'] = embeddings_norm
            kwargs['embeddings_weightnorm'] = True

        if cfg.MODEL.VISUAL_EMBED.LOCATION_SIZE > 0:
            embeddings_pos = nn.Linear(5, cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.ATT_FEATS]
        boxes = batched_inputs[kfg.ATT_FEATS_LOC] if kfg.ATT_FEATS_LOC in batched_inputs else None

        embeddings = self.embeddings(feats)
        if (self.embeddings_pos is not None) and (boxes is not None):
            embeddings_pos = self.embeddings_pos(boxes)
            embeddings = embeddings + embeddings_pos

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(inputs=embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        return { kfg.ATT_FEATS: embeddings }