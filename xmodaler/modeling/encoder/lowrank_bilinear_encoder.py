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
from .build import ENCODER_REGISTRY
from xmodaler.modeling.layers import LowRankBilinearLayer
from xmodaler.modeling.layers import get_act_layer

__all__ = ["LowRankBilinearEncoder"]

@ENCODER_REGISTRY.register()
class LowRankBilinearEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        embed_dim: int, 
        att_heads: int,
        att_mid_dim: int,
        att_mid_drop: float,
        dropout: float, 
        bifeat_emb_dropout: float,
        layer_num: int,
        emb_act_type: str,
        act_type: str,
        elu_alpha: float
    ):
        super(LowRankBilinearEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.bifeat_emb = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([]) 
        for _ in range(layer_num):
            sublayer = LowRankBilinearLayer( 
                embed_dim = embed_dim,
                att_heads = att_heads,
                att_mid_dim = att_mid_dim,
                att_mid_drop = att_mid_drop,
                dropout = dropout,
                act_type= act_type,
                elu_alpha = elu_alpha
            )
            self.layers.append(sublayer)

            self.bifeat_emb.append(nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                get_act_layer(emb_act_type)(),
                nn.Dropout(bifeat_emb_dropout)
            ))

            self.layer_norms.append(torch.nn.LayerNorm(embed_dim))

        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.BILINEAR.DIM,
            "att_heads": cfg.MODEL.BILINEAR.HEAD,
            "att_mid_dim": cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DIM,
            "att_mid_drop": cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DROPOUT,
            "dropout": cfg.MODEL.BILINEAR.ENCODE.DROPOUT,
            "bifeat_emb_dropout": cfg.MODEL.BILINEAR.ENCODE.BIFEAT_EMB_DROPOUT,
            "layer_num": cfg.MODEL.BILINEAR.ENCODE.LAYERS,
            "emb_act_type": cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT,
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

        cfg.MODEL.BILINEAR.ENCODE = CN()
        cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DIM = [128, 64, 128]
        cfg.MODEL.BILINEAR.ENCODE.ATT_MID_DROPOUT = 0.1
        cfg.MODEL.BILINEAR.ENCODE.DROPOUT = 0.5
        cfg.MODEL.BILINEAR.ENCODE.BIFEAT_EMB_DROPOUT = 0.3
        cfg.MODEL.BILINEAR.ENCODE.LAYERS = 4

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            att_feats = batched_inputs[kfg.ATT_FEATS]
            att_mask = batched_inputs[kfg.ATT_MASKS]
            # global feats
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            
            feat_arr = [gv_feat]
            for i, layer in enumerate(self.layers):
                gv_feat = layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
                att_feats_cat = torch.cat([gv_feat.unsqueeze(1).expand_as(att_feats), att_feats], dim = -1)

                att_feats = self.bifeat_emb[i](att_feats_cat) + att_feats
                att_feats = self.layer_norms[i](att_feats)
                feat_arr.append(gv_feat)

            gv_feat = torch.cat(feat_arr, dim=-1)
            gv_feat = self.proj(gv_feat)
            gv_feat = self.layer_norm(gv_feat)

            ret.update({ kfg.ATT_FEATS: att_feats, kfg.GLOBAL_FEATS: gv_feat })

        return ret