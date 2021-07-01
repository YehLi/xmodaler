import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.soft_attention import SoftAttention
from .rnn_decoder import RnnDecoder
from .build import DECODER_REGISTRY

__all__ = ["MPLSTMDecoder"]

@DECODER_REGISTRY.register()
class MPLSTMDecoder(RnnDecoder):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        token_embed_dim: int,
        visual_embed_dim: int,
        lstm_input_dropout: float
    ):
        super(MPLSTMDecoder, self).__init__()

        self.num_layers = 1
        self.hidden_size = hidden_size

        in_dim = token_embed_dim + visual_embed_dim
        self.lstm = nn.LSTMCell(in_dim, hidden_size)
        self.lstm_input_dropout = nn.Dropout(lstm_input_dropout) if lstm_input_dropout > 0 else None

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "token_embed_dim": cfg.MODEL.TOKEN_EMBED.DIM,
            "visual_embed_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "lstm_input_dropout": cfg.MODEL.MPLSTM.LSTM_INPUT_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.MPLSTM = CN()
        cfg.MODEL.MPLSTM.LSTM_INPUT_DROPOUT = 0.5 
        cfg.MODEL.MPLSTM.LM_DROPOUT = 0.5

    def preprocess(self, batched_inputs):
        att_feats = batched_inputs[kfg.ATT_FEATS]
        init_states = self.init_states(att_feats.shape[0])
        
        batched_inputs.update(init_states)
        return batched_inputs

    def forward(self, batched_inputs):
        xt = batched_inputs[kfg.TOKEN_EMBED]
        gv_feat = batched_inputs[kfg.GLOBAL_FEATS]
        hidden_states = batched_inputs[kfg.HIDDEN_STATES] # [num_layer, batch_size, hidden_size]
        cell_states = batched_inputs[kfg.CELL_STATES]

        input_combined = torch.cat((xt, gv_feat), dim=-1) # [batch_size, 3072]

        if self.lstm_input_dropout:
            input_combined = self.lstm_input_dropout(input_combined)

        h, c = self.lstm(input_combined, (hidden_states[0], cell_states[0]))

        return { 
                    kfg.HIDDEN_STATES: h.unsqueeze(0),
                    kfg.CELL_STATES: c.unsqueeze(0),
        }
    
