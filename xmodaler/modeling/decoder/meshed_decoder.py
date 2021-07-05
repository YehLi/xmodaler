import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.soft_attention import SoftAttention
from ..layers.multihead_attention import MultiHeadAttention
from ..layers.positionwise_feedforward import PositionWiseFeedForward

from .decoder import Decoder
from .build import DECODER_REGISTRY

import numpy as np 
import sys 

__all__ = ["MeshedDecoder"]

class MeshedDecoderLayer(nn.Module):
    def __init__(
        self, 
        *,
        d_model=512, 
        num_head=8, 
        d_ff=2048, 
        dropout=.1
    ):
        super(MeshedDecoderLayer, self).__init__()

        d_k = d_v = d_model // num_head

        self.self_att = MultiHeadAttention( d_model=d_model, 
                                            d_k=d_k, 
                                            d_v=d_v, 
                                            num_head=num_head, 
                                            dropout=dropout, 
                                            can_be_stateful=True
                                        )
        self.enc_att = MultiHeadAttention(  d_model=d_model, 
                                            d_k=d_k, 
                                            d_v=d_v, 
                                            num_head=num_head, 
                                            dropout=dropout, 
                                            can_be_stateful=False
                                        )
                                        
        self.pwff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        # cal attention on each encoder layer
        enc_att1 = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att) * mask_pad
        enc_att2 = self.enc_att(self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att) * mask_pad
        enc_att3 = self.enc_att(self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att) * mask_pad

        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))

        # weighted sum
        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3) / np.sqrt(3)
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff

@DECODER_REGISTRY.register()
class MeshedDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        d_model: int , 
        num_layer: int,  
        num_att_head: int, 
        d_ff: int, 
        dropout: float,
        padding_idx: int # -1
    ):
        super(MeshedDecoder, self).__init__()

        self.num_layers = num_layer
        self.d_model = d_model
        self.num_att_head = num_att_head
        self.d_ff = d_ff
        self.dropout = dropout
        self.padding_idx = padding_idx

        d_k = d_v = self.d_model // self.num_att_head

        self.layers = nn.ModuleList(
            [MeshedDecoderLayer(
                                d_model=self.d_model, 
                                num_head=self.num_att_head, 
                                d_ff=self.d_ff, 
                                dropout=self.dropout) for _ in range(self.num_layers)])

        # self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        # self.register_state('running_seq', torch.zeros((1,)).long())

    @classmethod
    def from_config(cls, cfg):
        return {
            "d_model": cfg.MODEL.MESHEDMEORY.DECODER.DIM_MODEL,
            "num_layer": cfg.MODEL.MESHEDMEORY.DECODER.NUM_LAYER,
            "num_att_head": cfg.MODEL.MESHEDMEORY.DECODER.NUM_ATT_HEAD,
            "d_ff": cfg.MODEL.MESHEDMEORY.DECODER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.MESHEDMEORY.DECODER.DROPOUT,
            "padding_idx": -1 # default
        }

    @classmethod
    def add_config(cls, cfg):
        if not hasattr(cfg.MODEL, "MESHEDMEORY"):
            cfg.MODEL.MESHEDMEORY = CN()

        cfg.MODEL.MESHEDMEORY.DECODER = CN()
        cfg.MODEL.MESHEDMEORY.DECODER.DIM_MODEL = 512
        cfg.MODEL.MESHEDMEORY.DECODER.NUM_LAYER = 3
        cfg.MODEL.MESHEDMEORY.DECODER.DROPOUT = 0.1
        cfg.MODEL.MESHEDMEORY.DECODER.NUM_ATT_HEAD = 8
        cfg.MODEL.MESHEDMEORY.DECODER.DIM_FEEDFORWARD = 2048

        kfg.SELF_ATT_MASKS = 'SELF_ATT_MASKS'
        kfg.SEQ_MASKS = 'SEQ_MASKS'

    def preprocess(self, batched_inputs):
        return {} 

    def forward(self, batched_inputs):
        input = batched_inputs[kfg.TOKEN_EMBED]
        mask_self_attention = batched_inputs[kfg.SELF_ATT_MASKS]
        mask_queries = batched_inputs[kfg.SEQ_MASKS]
        encoder_output = batched_inputs[kfg.ATT_FEATS]
        mask_encoder = batched_inputs[kfg.ATT_MASKS] # original mask
        
        # if self._is_stateful:
        #     self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
        #     mask_self_attention = self.running_mask_self_attention
        
        # seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        # seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        # if self._is_stateful:
        #     self.running_seq.add_(1)
        #     seq = self.running_seq
        
        out = input
        
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        return out 