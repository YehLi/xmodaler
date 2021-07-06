import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .decoder import Decoder
from .build import DECODER_REGISTRY

__all__ = ["MPLSTMDecoder"]

@DECODER_REGISTRY.register()
class MPLSTMDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        token_embed_dim: int,
    ):
        super(MPLSTMDecoder, self).__init__()
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(token_embed_dim, hidden_size)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "token_embed_dim": cfg.MODEL.TOKEN_EMBED.DIM,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def preprocess(self, batched_inputs):
        gv_feat = batched_inputs[kfg.GLOBAL_FEATS]
        init_states = self.init_states(gv_feat.shape[0])
        hidden_state, cell_state = self.lstm(gv_feat, 
            (init_states[kfg.G_HIDDEN_STATES][0], init_states[kfg.G_CELL_STATES][0]))
        
        batched_inputs.update({
            kfg.G_HIDDEN_STATES: [hidden_state],
            kfg.G_CELL_STATES: [cell_state]
        })
        return batched_inputs

    def forward(self, batched_inputs):
        xt = batched_inputs[kfg.G_TOKEN_EMBED]
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES] # [num_layer, batch_size, hidden_size]
        cell_states = batched_inputs[kfg.G_CELL_STATES]

        hidden_state, cell_state = self.lstm(xt, (hidden_states[0], cell_states[0]))

        return { 
            kfg.G_HIDDEN_STATES: [hidden_state],
            kfg.G_CELL_STATES: [cell_state]
        }
    
