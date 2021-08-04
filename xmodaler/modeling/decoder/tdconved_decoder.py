# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jingwen Chen
@contact: chenjingwen.sysu@gmail.com
"""
import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers import ShiftedConvLayer, SoftAttention
from .decoder import Decoder
from .build import DECODER_REGISTRY
import math 

__all__ = ["TDConvEDDecoder"]

@DECODER_REGISTRY.register()
class TDConvEDDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        hidden_size: int,
        kernel_sizes: list, # list of int
        conv_dropout: float,
        att_embed_size: int, 
        att_embed_dropout: float,
        use_norm: bool
    ):
        super(TDConvEDDecoder, self).__init__()
        self.num_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.conv_dropout = conv_dropout
        self.att_embed_size = att_embed_size
        self.att_embed_dropout = att_embed_dropout

        if use_norm:
            self.gv_feat_embed = weight_norm(nn.Linear(hidden_size, hidden_size))
            self.gv_feat_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0. else None

            self.wt_gv_embed = weight_norm(nn.Linear(hidden_size * 2, hidden_size))
            self.wt_gv_embed_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0. else None
        
            self.p_att_feats = weight_norm(nn.Linear(hidden_size, att_embed_size))
            self.p_att_feats_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0. else None
        else:
            self.gv_feat_embed = nn.Linear(hidden_size, hidden_size)
            self.gv_feat_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0. else None

            self.wt_gv_embed = nn.Linear(hidden_size * 2, hidden_size)
            self.wt_gv_embed_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0. else None

            self.p_att_feats = nn.Linear(hidden_size, att_embed_size)
            self.p_att_feats_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0. else None
        
        self.layers = nn.ModuleList(
            [ShiftedConvLayer(
                hidden_size,
                hidden_size,
                kernel_size, # list of int 
                stride=1,
                padding_mode='zeros', # 'zeros'
                dropout=conv_dropout,
                use_norm=use_norm) for kernel_size in self.kernel_sizes]
        )

        self.att = SoftAttention(
            hidden_size = hidden_size,
            att_embed_size = att_embed_size,
            att_embed_dropout = att_embed_dropout,
            use_norm = use_norm
        )

        self._clear_decoding_buffer()

        
    @classmethod
    def from_config(cls, cfg):
        return {
            "num_hidden_layers": cfg.MODEL.TDCONVED.DECODER.NUM_HIDDEN_LAYERS,
            "hidden_size": cfg.MODEL.TDCONVED.DECODER.HIDDEN_SIZE,
            "kernel_sizes": cfg.MODEL.TDCONVED.DECODER.KERNEL_SIZES, # list of int
            "conv_dropout": cfg.MODEL.TDCONVED.DECODER.DROPOUT,
            "att_embed_size": cfg.MODEL.TDCONVED.DECODER.ATT_EMBED_SIZE, 
            "att_embed_dropout": cfg.MODEL.TDCONVED.DECODER.ATT_EMBED_DROPOUT,
            "use_norm": cfg.MODEL.TDCONVED.DECODER.USE_NORM
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.TDCONVED.DECODER = CN()
        cfg.MODEL.TDCONVED.DECODER.NUM_HIDDEN_LAYERS = 2
        cfg.MODEL.TDCONVED.DECODER.HIDDEN_SIZE = 512
        cfg.MODEL.TDCONVED.DECODER.KERNEL_SIZES = [3, 3]
        cfg.MODEL.TDCONVED.DECODER.DROPOUT = 0.5
        cfg.MODEL.TDCONVED.DECODER.ATT_EMBED_SIZE = 256
        cfg.MODEL.TDCONVED.DECODER.ATT_EMBED_DROPOUT = 0.5
        cfg.MODEL.TDCONVED.DECODER.USE_NORM = True 

    def preprocess(self, batched_inputs):
        att_feats = batched_inputs[kfg.ATT_FEATS]
        batch_size, num_frames, hidden_size = att_feats.size()
        att_masks = batched_inputs[kfg.ATT_MASKS].view(batch_size, num_frames) # [batch, num_frames]
        ext_att_masks = batched_inputs[kfg.EXT_ATT_MASKS] # 4-D

        p_att_feats = self.p_att_feats(att_feats)
        if self.p_att_feats_dropout is not None:
            p_att_feats = self.p_att_feats_dropout(p_att_feats)

        gv_feat = torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1)
        gv_feat = self.gv_feat_embed(gv_feat)
        if self.gv_feat_dropout is not None:
            gv_feat = self.gv_feat_dropout(gv_feat)

        if self.training:
            self._clear_decoding_buffer()
            wt = batched_inputs[kfg.G_TOKENS_IDS] # [batch, max_len]
            seq_len = wt.size(1)
            # expand along time
            batched_inputs.update( {    
                                    kfg.P_ATT_FEATS:    p_att_feats.unsqueeze(1).expand(batch_size, seq_len, num_frames, self.att_embed_size)
                                                        .contiguous().view(-1, num_frames, self.att_embed_size), 
                                    kfg.GLOBAL_FEATS:   gv_feat.unsqueeze(1).expand(batch_size, seq_len, hidden_size),
                                    kfg.ATT_FEATS:      att_feats.unsqueeze(1).expand(batch_size, seq_len, num_frames, hidden_size)
                                                        .contiguous().view(-1, num_frames, hidden_size),
                                    kfg.EXT_ATT_MASKS:  ext_att_masks.expand(batch_size, seq_len, 1, num_frames)
                                                        .contiguous().view(-1, num_frames)
                                    } )
        else:
            self._init_decoding_buffer(batch_size)
            wt = batched_inputs[kfg.G_TOKENS_TYPE] # [batch, max_len]
            seq_len = wt.size(1)
            # expand along time
            batched_inputs.update( {    
                                    kfg.P_ATT_FEATS:    p_att_feats.unsqueeze(1).expand(batch_size, seq_len, num_frames, self.att_embed_size), 
                                    kfg.GLOBAL_FEATS:   gv_feat.unsqueeze(1).expand(batch_size, seq_len, hidden_size),
                                    kfg.ATT_FEATS:      att_feats.unsqueeze(1).expand(batch_size, seq_len, num_frames, hidden_size),
                                    kfg.EXT_ATT_MASKS:  ext_att_masks.expand(batch_size, seq_len, 1, num_frames)
                                    } )
        '''                            
        batched_inputs.update( {    kfg.P_ATT_FEATS: p_att_feats.unsqueeze(1).tile(1, seq_len, 1, 1).view(-1, num_frames, self.att_embed_size), 
                                    kfg.GLOBAL_FEATS: gv_feat.unsqueeze(1).tile(1, seq_len, 1),
                                    kfg.ATT_FEATS: att_feats.unsqueeze(1).tile(1, seq_len, 1, 1).view(-1, num_frames, dimension),
                                    kfg.EXT_ATT_MASKS: ext_att_masks.tile(1, seq_len, 1, 1).view(-1, num_frames)
                                    } )
        '''
        return batched_inputs

    def _init_decoding_buffer(self, batch_size):
        self.pred_token_embed = torch.zeros(batch_size, 0, self.hidden_size, dtype=torch.long).cuda()

    def _clear_decoding_buffer(self):
        self.pred_token_embed = None
    
    def forward(self, batched_inputs):
        wt = batched_inputs[kfg.G_TOKEN_EMBED]
        att_feats = batched_inputs[kfg.ATT_FEATS]
        ext_att_masks = batched_inputs[kfg.EXT_ATT_MASKS]
        p_att_feats = batched_inputs[kfg.P_ATT_FEATS]
        global_feats = batched_inputs[kfg.GLOBAL_FEATS] 
        history_states = batched_inputs.get(kfg.HISTORY_STATES, None)
        
        if self.training:
            cur_input_embed = torch.cat([wt, global_feats], axis=-1)
            cur_att_feats = att_feats
            cur_att_masks = ext_att_masks
            cur_p_att_feats = p_att_feats
            history_states = [None] * (self.num_layers + 1)
        else:
            time_step = batched_inputs[kfg.TIME_STEP]
            batch_size = att_feats.size(0)
            
            beam_size = wt.size(0) // batch_size
            if wt.dim() == 2: # [batch * beam, 1, hidden_size]
                wt = wt.unsqueeze(1)

            # init history_states
            if kfg.HISTORY_STATES not in batched_inputs:
                shape = list(wt.size()) # [batch * beam, 1, hidden_size]
                shape[1] = 0
                history_states = [wt.new(torch.Size(shape))] * (self.num_layers + 1) # additional one for input layer
                batched_inputs[kfg.HISTORY_STATES] = history_states

            # input of current time step
            max_seq_len, num_frames, hidden_size = att_feats.size(-3), att_feats.size(-2), att_feats.size(-1)
            cur_global_feats = (global_feats[:, time_step:time_step+1, :]).unsqueeze(1).expand(batch_size, beam_size, 1, hidden_size)
            cur_global_feats = cur_global_feats.view(-1, 1, hidden_size)
            cur_input_embed = torch.cat([wt, cur_global_feats], axis=-1)
            # [batch * beam * time, num_frames, hidden]
            cur_att_feats = (att_feats[:, :time_step+1, :, :]).unsqueeze(1).expand(batch_size, beam_size, time_step+1, num_frames, hidden_size) \
                            .contiguous().view(-1, num_frames, hidden_size)
            # [batch * beam * time, num_frames], -inf
            cur_att_masks = (ext_att_masks[:, :time_step+1, :, :]).unsqueeze(1).expand(batch_size, beam_size, time_step+1, 1, num_frames) \
                            .contiguous().view(-1, num_frames)
            # [batch * beam * time, num_frames, att_embed_size]
            cur_p_att_feats = (p_att_feats[:, :time_step+1, ]).unsqueeze(1).expand(batch_size, beam_size, time_step+1, num_frames, self.att_embed_size) \
                            .contiguous().view(-1, num_frames, self.att_embed_size)

        cur_input_embed = self.wt_gv_embed(cur_input_embed)
        if self.wt_gv_embed_dropout is not None:
            cur_input_embed = self.wt_gv_embed_dropout(cur_input_embed)
        if history_states[0] is not None: # for test
            input_embed = torch.cat([history_states[0], cur_input_embed], axis=1)
            history_states[0] = input_embed # update the history states
        else:
            input_embed = cur_input_embed
        
        layer_outputs = []
        layer_input = input_embed
        for idx, layer_module in enumerate(self.layers):
            layer_output = layer_module(layer_input)
            layer_output = (layer_output + layer_input) * math.sqrt(0.5) # residual connection
            layer_outputs.append(layer_output)
            if history_states[idx+1] is not None: # update the new hidden state for current step
                history_states[idx+1] = torch.cat([history_states[idx+1], layer_output[:, -1:, :]], axis=1)
            layer_input = layer_output
        
        # attention
        batch_size = layer_output.size(0)
        hidden_states = layer_output.view(-1, self.hidden_size) # [batch * beam * time_step, hidden_size]
        att_outputs = self.att(hidden_states, cur_att_feats, cur_p_att_feats, cur_att_masks)
        att_outputs = att_outputs.view(batch_size, -1, self.hidden_size)
        layer_output = (layer_output + att_outputs) * math.sqrt(0.5)
        
        if not self.training:
            return { 
                kfg.G_HIDDEN_STATES: layer_output[:, -1, :],
                kfg.HISTORY_STATES: history_states
            }
        else:
            return { 
                kfg.G_HIDDEN_STATES: layer_output
            }
