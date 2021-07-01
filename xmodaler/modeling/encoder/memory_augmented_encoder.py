import torch
from torch import nn
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .build import ENCODER_REGISTRY
from ..layers.positionwise_feedforward import PositionWiseFeedForward
from ..layers.multihead_attention import MultiHeadAttentionMemory

__all__ = ["MemoryAugmentedEncoder"]

class EncoderLayer(nn.Module):
    def __init__(
        self, 
        *,
        d_model=512,  
        num_head=8, 
        num_memory=40,
        d_ff=2048, 
        dropout=.1
    ):
        super(EncoderLayer, self).__init__()
        
        d_k = d_v = d_model // num_head

        self.mhatt = MultiHeadAttentionMemory(  d_model=d_model, 
                                                d_k=d_k, 
                                                d_v=d_v, 
                                                num_head=num_head, 
                                                dropout=dropout, 
                                                num_memory=num_memory)

        self.pwff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, queries, keys, values, attention_mask):
        att = self.mhatt(queries, keys, values, attention_mask)
        ff = self.pwff(att)
        return ff

@ENCODER_REGISTRY.register()
class MemoryAugmentedEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        input_dim: int, # out_dim of visual embedding
        d_model: int,   # hidden size
        num_layer: int, 
        num_att_head: int, 
        num_att_memory: int, # memory attention 
        d_ff: int, # feedforward size
        dropout: float,
        padding_idx: int
    ):
        super(MemoryAugmentedEncoder, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layer
        self.num_att_head = num_att_head
        self.num_att_memory = num_att_memory
        self.d_ff = d_ff
        self.padding_idx = padding_idx
        self.dropout = dropout

        # encoder input layer
        self.fc = nn.Linear(self.input_dim, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout) if self.dropout > 0. else None
        self.layer_norm = nn.LayerNorm(self.d_model)

        d_v = d_k = self.d_model // self.num_att_head

        # encoder hidden layers
        self.layers = nn.ModuleList([EncoderLayer(  d_model=self.d_model,  
                                                    num_head=self.num_att_head, 
                                                    num_memory=self.num_att_memory,
                                                    d_ff=self.d_ff, 
                                                    dropout=dropout    
                                                    )
                                     for _ in range(self.num_layers)])

    @classmethod
    def from_config(cls, cfg):

        return {
            "input_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "d_model": cfg.MODEL.MESHEDMEORY.ENCODER.DIM_MODEL,
            "num_layer": cfg.MODEL.MESHEDMEORY.ENCODER.NUM_LAYER,
            "num_att_head": cfg.MODEL.MESHEDMEORY.ENCODER.NUM_ATT_HEAD,
            "num_att_memory": cfg.MODEL.MESHEDMEORY.ENCODER.NUM_ATT_MEMORY,
            "d_ff": cfg.MODEL.MESHEDMEORY.ENCODER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.MESHEDMEORY.ENCODER.DROPOUT,
            "padding_idx": 0 # default
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.MESHEDMEORY = CN()
        
        cfg.MODEL.MESHEDMEORY.ENCODER = CN()

        cfg.MODEL.MESHEDMEORY.ENCODER.DIM_MODEL = 512
        cfg.MODEL.MESHEDMEORY.ENCODER.NUM_LAYER = 3
        cfg.MODEL.MESHEDMEORY.ENCODER.DROPOUT = 0.1
        cfg.MODEL.MESHEDMEORY.ENCODER.NUM_ATT_HEAD = 8
        cfg.MODEL.MESHEDMEORY.ENCODER.NUM_ATT_MEMORY = 40
        cfg.MODEL.MESHEDMEORY.ENCODER.DIM_FEEDFORWARD = 2048

        # kfg.PREDICTOR_CTX = 'PREDICTOR_CTX'

    def _get_global_feat(self, feats, masks):
        if masks is None:
            global_feats = torch.mean(feats, 1)
        else:
            feats_masks = feats * masks.unsqueeze(-1)
            masks_sum = masks.sum(-1)
            global_feats = feats_masks.sum(1) / masks_sum.unsqueeze(-1)
        return global_feats

    def forward(self, batched_inputs):
        att_feats = batched_inputs[kfg.ATT_FEATS]
        att_masks = batched_inputs[kfg.ATT_MASKS]

        # input layer
        inputs = F.relu(self.fc(att_feats))
        if self.dropout:
            inputs = self.dropout(inputs)
        inputs = self.layer_norm(inputs)

        # running over encoder layers
        attention_mask = (torch.sum(inputs, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        global_feats = []
        out = inputs
        for l in self.layers:
            out = l(out, out, out, attention_mask)
            outs.append(out.unsqueeze(1))

            gv = self._get_global_feat(out, att_masks)
            global_feats.append(gv.unsqueeze(1))

        outs = torch.cat(outs, 1) # [batch, num_layer, seq_len, d_model]
        gv_feats = torch.cat(global_feats, 1) # [batch, num_layer, d_model]
        
        return {
            kfg.ATT_FEATS: outs,
            kfg.ATT_MASKS: attention_mask,
            kfg.GLOBAL_FEATS: gv_feats
        }