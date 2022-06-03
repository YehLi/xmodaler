# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.utils.initialization import trunc_normal_
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["VisualGridEmbedding"]

@EMBEDDING_REGISTRY.register()
class VisualGridEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        in_dim: int,
        g_in_dim: int,
        out_dim: int,
        **kwargs
    ):
        super(VisualGridEmbedding, self).__init__()
        self.embeddings = nn.Linear(in_dim, out_dim)
        self.g_embeddings = nn.Linear(g_in_dim, out_dim) if g_in_dim > 0 else None
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "in_dim": cfg.MODEL.VISUAL_EMBED.IN_DIM,
            "g_in_dim": cfg.MODEL.VISUAL_EMBED.G_IN_DIM,
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
            embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        embeddings_pos = nn.Parameter(
            torch.zeros(1, cfg.DATALOADER.MAX_FEAT_NUM, cfg.MODEL.VISUAL_EMBED.OUT_DIM))
        trunc_normal_(embeddings_pos, std=.02)
        kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.ATT_FEATS]
        embeddings = self.embeddings(feats)

        if self.g_embeddings is not None:
            gfeats = batched_inputs[kfg.GLOBAL_FEATS].view(embeddings.shape[0], -1, embeddings.shape[-1])
            g_embeddings = self.g_embeddings(gfeats)
            embeddings = torch.cat([g_embeddings, embeddings], dim=1)

        embeddings_pos = self.embeddings_pos
        embeddings = embeddings + embeddings_pos

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        return { kfg.ATT_FEATS: embeddings }