import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY
from .position_embedding import build_position_encoding

__all__ = ["TokenBaseEmbedding"]

@EMBEDDING_REGISTRY.register()
class TokenBaseEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        dim: int,
        vocab_size: int, # include <BOS>/<EOS>
        **kwargs
    ):
        super(TokenBaseEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop("embeddings_pos", None)
        self.embeddings_token_type = kwargs.pop('embeddings_token_type', None)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "dim": cfg.MODEL.TOKEN_EMBED.DIM, 
            "vocab_size": cfg.MODEL.VOCAB_SIZE
        }

        activation_name = (cfg.MODEL.TOKEN_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.TOKEN_EMBED.ELU_ALPHA
            embeddings_act = activation(act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.TOKEN_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.TOKEN_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.TOKEN_EMBED.USE_NORM:
            embeddings_norm = nn.LayerNorm(cfg.MODEL.TOKEN_EMBED.DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        if (cfg.MODEL.TOKEN_EMBED.POSITION).lower() != 'none':
            embeddings_pos = build_position_encoding(cfg,
                cfg.MODEL.TOKEN_EMBED.DIM, cfg.MODEL.TOKEN_EMBED.POSITION_MAX_LEN)
            kwargs['embeddings_pos'] = embeddings_pos

        if cfg.MODEL.TOKEN_EMBED.TYPE_VOCAB_SIZE > 0:
            embeddings_token_type = nn.Embedding(
                cfg.MODEL.TOKEN_EMBED.TYPE_VOCAB_SIZE, cfg.MODEL.TOKEN_EMBED.DIM)
            kwargs['embeddings_token_type'] = embeddings_token_type

        return kwargs
            

    def forward(self, batched_inputs):
        ret = {}
        if kfg.U_TOKENS_IDS in batched_inputs: 
            u_tokens_ids = batched_inputs[kfg.U_TOKENS_IDS]
            u_tokens_type = batched_inputs.get(kfg.U_TOKENS_TYPE, None)
            u_token_embed = self._forward(u_tokens_ids, token_type_ids=u_tokens_type)
            ret.update({ kfg.U_TOKEN_EMBED: u_token_embed })

        if kfg.G_TOKENS_IDS in batched_inputs:
            g_tokens_ids = batched_inputs[kfg.G_TOKENS_IDS]
            g_tokens_type = batched_inputs.get(kfg.G_TOKENS_TYPE, None)
            g_token_embed = self._forward(g_tokens_ids, token_type_ids=g_tokens_type)
            ret.update({ kfg.G_TOKEN_EMBED: g_token_embed })
        return ret

    def _forward(self, input_ids, token_type_ids=None):
        embeddings = self.embeddings(input_ids)
        
        if self.embeddings_pos is not None:
            position_embeddings = self.embeddings_pos(input_ids)
            embeddings = embeddings + position_embeddings

        if (self.embeddings_token_type is not None) and (token_type_ids is not None):
            embeddings_token_type = self.embeddings_token_type(token_type_ids)
            embeddings = embeddings + embeddings_token_type

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        return embeddings