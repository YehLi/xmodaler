import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.soft_attention import SoftAttention
from .rnn_decoder import RnnDecoder
from .build import DECODER_REGISTRY

__all__ = ["SALSTMDecoder"]

@DECODER_REGISTRY.register()
class SALSTMDecoder(RnnDecoder):
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

        self.attention = SoftAttention(            
            hidden_size=hidden_size, 
            att_feat_size=visual_embed_dim,
            att_embed_size=att_embed_size,
            att_embed_dropout=att_embed_dropout
        )

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
        cfg.MODEL.SALSTM.ATT_EMBED_SIZE = 256
        cfg.MODEL.SALSTM.CTX_DROPOUT = 0.5 
        cfg.MODEL.SALSTM.ATT_EMBED_DROPOUT = 0.0
        cfg.MODEL.SALSTM.LM_DROPOUT = 0.5

        kfg.PREDICTOR_CTX = 'PREDICTOR_CTX'

    def preprocess(self, batched_inputs):
        att_feats = batched_inputs[kfg.ATT_FEATS]
        init_states = self.init_states(att_feats.shape[0])
        
        batched_inputs.update(init_states)
        return batched_inputs

    def forward(self, batched_inputs):
        xt = batched_inputs[kfg.TOKEN_EMBED]
        att_feats = batched_inputs[kfg.ATT_FEATS]
        att_mask = batched_inputs[kfg.ATT_MASKS] # original mask
        hidden_states = batched_inputs[kfg.HIDDEN_STATES] # [num_layer, batch_size, hidden_size]
        cell_states = batched_inputs[kfg.CELL_STATES]
        
        feats, _ = self.attention(hidden_states[0], att_feats, att_mask) # [batch_size, v_dim]
        
        input_combined = torch.cat((xt, feats), dim=-1) # [batch_size, 3072]
        h, c = self.lstm(input_combined, (hidden_states[0], cell_states[0]))

        ctx = torch.cat((h, xt, feats), dim=-1)

        return { 
                    kfg.HIDDEN_STATES: h.unsqueeze(0),
                    kfg.CELL_STATES: c.unsqueeze(0),
                    kfg.PREDICTOR_CTX: ctx
        }
    