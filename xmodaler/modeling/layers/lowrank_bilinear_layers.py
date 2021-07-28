# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo, Jingwen Chen
@contact: jianjieluo.sysu@gmail.com, chenjingwen.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.modeling.layers import get_act_layer
from .scattention import SCAttention

__all__ = ["LowRankBilinearAttention", "LowRankBilinearLayer"]

class LowRank(nn.Module):
    def __init__(
        self, 
        *,
        embed_dim: int, 
        att_heads: int, 
        att_mid_dim: list, 
        att_mid_drop: float,
        act_type: str,
        elu_alpha: float
    ):
        super(LowRank, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = 2 * embed_dim if act_type == 'GLU' else embed_dim

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        # act = nn.CELU(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)

        self.attn_net = SCAttention(att_mid_dim, att_mid_drop)

    # query -- batch_size * qdim
    # value -- batch_size * att_num * vdim
    def forward(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2

        attn_map = q.unsqueeze(-2) * k
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn

    # query -- batch_size * seq_num * qdim
    # value -- batch_size * att_num * vdim
    def forward2(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        query = query.view(-1, query.size()[-1])
        value1 = value1.view(-1, value1.size()[-1])
        
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if self.buffer_keys is not None and self.buffer_value2 is not None:
                self.buffer_keys = torch.cat([self.buffer_keys, k], dim=2)
                self.buffer_value2 = torch.cat([self.buffer_value2, v2], dim=2)
                k = self.buffer_keys
                v2 = self.buffer_value2
        else:
            k = key
            v2 = value2
        
        attn_map = q.unsqueeze(-2) * k.unsqueeze(-3)
        attn = self.attn_net.forward(attn_map, mask, v1, v2).transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2):
        batch_size = value2.size()[0]
        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2

class LowRankBilinearLayer(nn.Module):
    def __init__(
        self,  
        *,       
        embed_dim: int, 
        att_heads: int,
        att_mid_dim: list,
        att_mid_drop: float,
        dropout: float,
        act_type: str, 
        elu_alpha: float
    ):
        super(LowRankBilinearLayer, self).__init__()
        self.encoder_attn = LowRank(
            embed_dim = embed_dim, 
            att_heads = att_heads, 
            att_mid_dim = att_mid_dim, 
            att_mid_drop = att_mid_drop,
            act_type = act_type,
            elu_alpha = elu_alpha
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self, 
        x, 
        key=None, 
        mask=None, 
        value1=None, 
        value2=None, 
        precompute=False
    ):    
        x = self.encoder_attn(
            query=x,
            key=key if key is not None else x,
            mask=mask,
            value1=value1 if value1 is not None else x,
            value2=value2 if value2 is not None else x,
            precompute=precompute
        )
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def precompute(self, key, value2):
        return self.encoder_attn.precompute(key, value2)


class LowRankBilinearAttention(nn.Module):
    def __init__(
        self, 
        *,
        embed_dim: int, 
        att_heads: int,
        att_mid_dim: list,
        att_mid_drop: float,
        dropout: float, 
        layer_num: int,
        act_type: str, 
        elu_alpha: float
    ):
        super(LowRankBilinearAttention, self).__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(layer_num):
            sublayer = LowRankBilinearLayer( 
                embed_dim = embed_dim, 
                att_heads = att_heads,
                att_mid_dim = att_mid_dim,
                att_mid_drop = att_mid_drop,
                dropout = dropout,
                act_type = act_type, 
                elu_alpha = elu_alpha
            )
            self.layers.append(sublayer)
        
        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def precompute(self, key, value2):
        keys = []
        value2s = []
        for layer in self.layers:
            k, v = layer.precompute(key, value2)
            keys.append(k)
            value2s.append(v)
        return torch.cat(keys, dim=-1), torch.cat(value2s, dim=-1)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None, precompute=False):
        if precompute == True:
            dim = p_att_feats.size()[-1]
            keys = p_att_feats.narrow(-1, 0, dim // 2)
            value2s = p_att_feats.narrow(-1, dim // 2, dim // 2)
            dim = keys.size()[-1] // len(self.layers)
    
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = (torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1))
            else:
                gv_feat = torch.mean(att_feats, 1)

        feat_arr = [gv_feat]
        for i, layer in enumerate(self.layers):
            key = keys.narrow(-1, i * dim, dim) if precompute else att_feats
            value2 = value2s.narrow(-1, i * dim, dim) if precompute else att_feats
                            
            gv_feat = layer(gv_feat, key, att_mask, gv_feat, value2, precompute)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats