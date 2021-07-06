import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.base_attention import BaseAttention
from .decoder import Decoder
from .build import DECODER_REGISTRY

__all__ = ["SALSTMDecoder"]

@DECODER_REGISTRY.register()
class SALSTMDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        token_embed_dim: int,
        visual_embed_dim: int,
        att_embed_size: int, 
        att_embed_dropout: float
    ):
        super(SALSTMDecoder, self).__init__()

        self.num_layers = 1
        self.hidden_size = hidden_size

        self.att = BaseAttention(
            hidden_size = hidden_size,
            att_embed_size = att_embed_size,
            att_embed_dropout = att_embed_dropout
        )
        self.p_att_feats = nn.Linear(hidden_size, att_embed_size)

        in_dim = token_embed_dim + visual_embed_dim
        self.lstm = nn.LSTMCell(in_dim, hidden_size)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "token_embed_dim": cfg.MODEL.TOKEN_EMBED.DIM,
            "visual_embed_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "att_embed_size": cfg.MODEL.SALSTM.ATT_EMBED_SIZE,
            "att_embed_dropout": cfg.MODEL.SALSTM.ATT_EMBED_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.SALSTM = CN()
        cfg.MODEL.SALSTM.ATT_EMBED_SIZE = 512
        cfg.MODEL.SALSTM.ATT_EMBED_DROPOUT = 0.0

    def preprocess(self, batched_inputs):
        att_feats = batched_inputs[kfg.ATT_FEATS]
        p_att_feats = self.p_att_feats(att_feats)
        init_states = self.init_states(att_feats.shape[0])
        
        batched_inputs.update(init_states)
        batched_inputs.update( { kfg.P_ATT_FEATS: p_att_feats } )
        return batched_inputs

    def forward(self, batched_inputs):
        wt = batched_inputs[kfg.G_TOKEN_EMBED]
        att_feats = batched_inputs[kfg.ATT_FEATS]
        ext_att_masks = batched_inputs[kfg.EXT_ATT_MASKS]
        p_att_feats = batched_inputs[kfg.P_ATT_FEATS]
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
        cell_states = batched_inputs[kfg.G_CELL_STATES]

        att = self.att(hidden_states[0], att_feats, p_att_feats, ext_att_masks)
        input_combined = torch.cat((wt, att), dim=-1)
        hidden_state, cell_state = self.lstm(input_combined, (hidden_states[0], cell_states[0]))

        return { 
            kfg.G_HIDDEN_STATES: [hidden_state],
            kfg.G_CELL_STATES: [cell_state],
        }