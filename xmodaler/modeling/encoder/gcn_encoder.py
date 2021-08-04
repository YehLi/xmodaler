# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .build import ENCODER_REGISTRY

__all__ = ["GCNEncoder"]

@ENCODER_REGISTRY.register()
class GCNEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        relation_num: int,
        dropout: float, 
    ):
        super(GCNEncoder, self).__init__()
        self.gcn_loop = nn.Linear(in_dim, out_dim)
        self.gcn_sub = nn.Linear(in_dim, out_dim, bias=False)
        self.gcn_obj = nn.Linear(in_dim, out_dim, bias=False)

        self.gcn_loop_gate = nn.Linear(in_dim, 1)
        self.gcn_sub_gate = nn.Linear(in_dim, 1, bias=False)
        self.gcn_obj_gate = nn.Linear(in_dim, 1, bias=False)

        self.gcn_loop_gate_act = nn.Sigmoid()
        self.gcn_sub_gate_act = nn.Sigmoid()
        self.gcn_obj_gate_act = nn.Sigmoid()

        self.gate_bias = nn.Parameter(torch.zeros(relation_num, 1).cuda(), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(relation_num, out_dim).cuda(), requires_grad=True)

        self.gcn_act = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "in_dim": cfg.MODEL.GCN.IN_DIM,
            "out_dim": cfg.MODEL.GCN.OUT_DIM,
            "relation_num": cfg.MODEL.GCN.RELATION_NUM,
            "dropout": cfg.MODEL.GCN.DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.GCN = CN()
        cfg.MODEL.GCN.IN_DIM = 2048
        cfg.MODEL.GCN.OUT_DIM = 2048
        cfg.MODEL.GCN.RELATION_NUM = 21
        cfg.MODEL.GCN.DROPOUT = 0.5

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode != 'v':
            return ret

        att_feats = batched_inputs[kfg.ATT_FEATS]
        # att_masks = batched_inputs[kfg.ATT_MASKS] # 36 features per image
        gcn_loop = self.gcn_loop(att_feats)
        gcn_loop_gate = self.gcn_loop_gate(att_feats)
        gcn_loop_gate = self.gcn_loop_gate_act(gcn_loop_gate)
        gcn_loop = gcn_loop * gcn_loop_gate

        gcn_sub = self.gcn_sub(att_feats)
        gcn_sub_gate = self.gcn_sub_gate(att_feats)

        gcn_obj = self.gcn_obj(att_feats)
        gcn_obj_gate = self.gcn_obj_gate(att_feats)

        rel = batched_inputs[kfg.RELATION]
        rel_bias = self.bias[rel]
        rel_gate_bias = self.gate_bias[rel]
        rel_mask = (rel > 0).to(dtype=rel.dtype).unsqueeze(-1)

        # v_i-to-v_j
        gcn_sub_gate = self.gcn_sub_gate_act(gcn_sub_gate.unsqueeze(1) + rel_gate_bias)
        gcn_sub = (gcn_sub.unsqueeze(1) + rel_bias)
        gcn_sub = (gcn_sub * gcn_sub_gate * rel_mask).sum(2)

        # v_j-to-v_i
        gcn_obj_gate = self.gcn_obj_gate_act(gcn_obj_gate.unsqueeze(2) + rel_gate_bias)
        gcn_obj = (gcn_obj.unsqueeze(2) + rel_bias)
        gcn_obj = (gcn_obj * gcn_obj_gate * rel_mask).sum(1)

        gcn_feat = gcn_loop + gcn_sub + gcn_obj + att_feats
        gcn_feat = self.gcn_act(gcn_feat)
        global_feats = torch.mean(gcn_feat, 1)
        ret.update({ 
            kfg.GLOBAL_FEATS: global_feats,
            kfg.ATT_FEATS: gcn_feat
        })
        return ret