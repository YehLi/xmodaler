# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo, Jingwen Chen
@contact: jianjieluo.sysu@gmail.com, chenjingwen.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from xmodaler.modeling.layers import LowRankBilinearAttention
from .build import DECODER_REGISTRY
from .decoder import Decoder

__all__ = ["XLANDecoder"]

@DECODER_REGISTRY.register()
class XLANDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        ctx_drop: float,
        bilinear_dim: int,
        att_heads: int, 
        att_mid_dim: int, 
        att_mid_drop: float,
        att_embed_dropout: float,
        layer_num: int,
        act_type: str,
        elu_alpha: float
    ):
        super(XLANDecoder, self).__init__()

        self.num_layers = 2
        self.hidden_size = hidden_size

        # First LSTM layer
        rnn_input_size = hidden_size + bilinear_dim
        self.att_lstm = nn.LSTMCell(rnn_input_size, hidden_size)
        self.ctx_drop = nn.Dropout(ctx_drop)

        # lowrank dec block
        self.attention = LowRankBilinearAttention( 
            embed_dim = bilinear_dim, 
            att_heads = att_heads,
            att_mid_dim = att_mid_dim,
            att_mid_drop = att_mid_drop,
            dropout = att_embed_dropout, 
            layer_num = layer_num,
            act_type = act_type,
            elu_alpha = elu_alpha
        )
        self.att2ctx = nn.Sequential(
            nn.Linear(bilinear_dim + hidden_size, 2 * hidden_size), 
            nn.GLU()
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "ctx_drop": cfg.MODEL.PRED_DROPOUT,
            "bilinear_dim": cfg.MODEL.BILINEAR.DIM,
            "att_heads": cfg.MODEL.BILINEAR.HEAD,
            "att_mid_dim": cfg.MODEL.BILINEAR.DECODE.ATT_MID_DIM,
            "att_mid_drop": cfg.MODEL.BILINEAR.DECODE.ATT_MID_DROPOUT,
            "att_embed_dropout":  cfg.MODEL.BILINEAR.DECODE.DROPOUT,
            "layer_num": cfg.MODEL.BILINEAR.DECODE.LAYERS,
            "act_type": cfg.MODEL.BILINEAR.ACT,
            "elu_alpha": cfg.MODEL.BILINEAR.ELU_ALPHA
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.BILINEAR = CN()
        cfg.MODEL.BILINEAR.DIM = 1024
        cfg.MODEL.BILINEAR.HEAD = 8
        cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT = "relu"
        cfg.MODEL.BILINEAR.ACT = "celu"
        cfg.MODEL.BILINEAR.ELU_ALPHA = 1.3

        cfg.MODEL.BILINEAR.DECODE = CN()
        cfg.MODEL.BILINEAR.DECODE.ATT_MID_DIM = [128, 64, 128]
        cfg.MODEL.BILINEAR.DECODE.ATT_MID_DROPOUT = 0.1
        cfg.MODEL.BILINEAR.DECODE.DROPOUT = 0.5
        cfg.MODEL.BILINEAR.DECODE.LAYERS = 1

        cfg.MODEL.BILINEAR.ENCODE = CN()
        cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DIM = [128, 64, 128]
        cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DROPOUT = 0.1
        cfg.MODEL.BILINEAR.ENCODE.DROPOUT = 0.5
        cfg.MODEL.BILINEAR.ENCODE.BIFEAT_EMB_DROPOUT = 0.3
        cfg.MODEL.BILINEAR.ENCODE.LAYERS = 4

    def preprocess(self, batched_inputs):
        att_feats = batched_inputs[kfg.ATT_FEATS]
        keys, value2s = self.attention.precompute(att_feats, att_feats)
        p_att_feats = torch.cat([keys, value2s], dim=-1)

        init_states = self.init_states(att_feats.shape[0])

        batched_inputs.update(init_states)
        batched_inputs.update({ kfg.P_ATT_FEATS: p_att_feats })
        return batched_inputs

    def forward(self, batched_inputs):
        wt = batched_inputs[kfg.G_TOKEN_EMBED]
        att_feats = batched_inputs[kfg.ATT_FEATS]
        att_masks = batched_inputs[kfg.ATT_MASKS]
        p_att_feats = batched_inputs[kfg.P_ATT_FEATS]
        gv_feat = batched_inputs[kfg.GLOBAL_FEATS]
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES] # list of tensors
        cell_states = batched_inputs[kfg.G_CELL_STATES]
        
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_masks is not None:
                gv_feat = torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        
        h_att, c_att = self.att_lstm(torch.cat([wt, gv_feat + self.ctx_drop(hidden_states[1])], 1), (hidden_states[0], cell_states[0]))
        att, _ = self.attention(h_att, att_feats, att_masks, p_att_feats, precompute=True)
        ctx_input = torch.cat([att, h_att], 1)

        output = self.att2ctx(ctx_input)
        hidden_states = [h_att, output]
        cell_states = [c_att, cell_states[1]]
        
        return { 
            kfg.G_HIDDEN_STATES: hidden_states,
            kfg.G_CELL_STATES: cell_states
        }