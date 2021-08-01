# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jingwen Chen
@contact: chenjingwen.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.modeling.layers import get_act_layer
from .scattention import SCAttention
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

__all__ = ["TemporalDeformableLayer", "ShiftedConvLayer", "SoftAttention"]

class TemporalDeformableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        # padding: int,
        padding_mode: str, # 'border'
        offset_act: str,
        min_idx: int, # 0
        max_idx: int,  # max_len
        clamp_idx: bool, # True
        use_norm: bool 
        ):

        super(TemporalDeformableBlock, self).__init__()
        
        self.kernel_size = kernel_size
        self.half_span = self.kernel_size // 2
        self.padding_mode = padding_mode
        self.padding = 0
        self.dilation = 1
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.clamp_idx = clamp_idx
        # offset conv : to compute the offsets
        if use_norm:
            self.offset_conv = weight_norm(nn.Conv1d(
                                        in_channels,
                                        kernel_size,
                                        kernel_size,
                                        stride,
                                        kernel_size // 2 # with padding
                                    )).cuda()
        else:
            self.offset_conv = nn.Conv1d(
                                        in_channels,
                                        kernel_size,
                                        kernel_size,
                                        stride,
                                        kernel_size // 2 # with padding
                                    ).cuda()

        self.offset_act = get_act_layer(offset_act)() if offset_act.lower() != "none" else None

        # the second conv : to compute outputs
        if use_norm:
            self.conv = weight_norm(nn.Conv1d(
                                in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                0 # without padding
                            )).cuda()
        else:
            self.conv = nn.Conv1d(
                                in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                0 # without padding
                            ).cuda() 
        # zero initialization
        self._init_offset_param()

    def _init_offset_param(self):
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    # inputs -- [batch_size, cfg.MODEL.TDCONVED.ENCODER_HIDDEN_DIM, TIME_STEP]
    def forward(self, inputs):
        # max_seq_len = inputs.size(-1) # [b, dim, max_len]
        # padded_inputs = self._padding_inputs(inputs) # [b, dim, TIME_STEP]
        padded_inputs = inputs

        # predict offsets
        offests = self.offset_conv(padded_inputs) # [batch, kernel_size, TIME_STEP]
        if self.offset_act is not None:
            offests = self.offset_act(offests)
        offests_pred = offests.permute(0, 2, 1).cpu() # [batch, TIME_STEP, kernel_size]

        sampling_positions = self._get_sampling_shift() # [kernel_size]

        # [batch, 1, max_len, dim]
        new_inputs = inputs.permute(0, 2, 1).unsqueeze(1) # [b, 1, TIME_STEP, ENCODER_HIDDEN_DIM]
        batch_size, _, H, W = new_inputs.size() # H -> MAX_LEN
        single_step_offset = self._normalize_offset(1.0, H) 

        sampled_feats = []
        for sample_idx, pos in enumerate(sampling_positions):
            # same sampling choice along time, each pos will have the same shift like -1 in [-1, 0, 1] for k =3
            # [batch, H, 1]
            cur_pos_offset = torch.Tensor([pos * single_step_offset])\
                                                .unsqueeze(0).repeat(H, 1)\
                                                .unsqueeze(0).repeat(batch_size, 1, 1)
            # the predicted offsets for the first sampling pos
            cur_pred_offset = offests_pred[:, :, sample_idx:sample_idx+1] 
            new_offset = cur_pos_offset + cur_pred_offset
            grid = self._make_grid(H, W, new_offset, batch_size) # temporal shift 1 step left, [b, h, w, 2]

            # [batch_size, 1, H, W]
            sampled_feat = F.grid_sample(   new_inputs, 
                                            grid, 
                                            mode="bilinear", 
                                            padding_mode=self.padding_mode
                                        )

            sampled_feats.append(sampled_feat.squeeze())
        
        # [batch_size * H, W, kernel_size]
        sampled_feats = torch.stack(sampled_feats, dim=-1).view(-1, W, self.kernel_size)

        # [b, h, 2 * w, k] -> [b * h, 2 * w, k] -> [b, h, 2 * w]
        outputs = self.conv(sampled_feats).view(batch_size, H, -1)
        # if self.padding_mode == "null":
        #    outputs = outputs[:, self.half_span:-self.half_span, :]
        #    assert outpus.size(1) == max_len - 2 * self.half_span
        return outputs

    def _padding_inputs(self, inputs):
        if self.padding_mode == "border":
            left_most_feat = inputs[:, :, 0:1].repeat(1, 1, self.half_span)
            right_most_feat = inputs[:, :, -1:].repeat(1, 1, self.half_span)
        elif self.padding_mode == "zeros" or self.padding_mode == "null":
            batch_size, W, H = inputs.size()
            left_most_feat = torch.zeros((batch_size, W, self.half_span)).float().cuda()
            right_most_feat = torch.zeros((batch_size, W, self.half_span)).float().cuda()
        else:
            raise NotImplementedError("not supported padding type")

        # if (isinstance(self.padding, int) and self.padding > 0) or self.padding_mode == "null":
        outputs = torch.cat((left_most_feat, inputs), dim=-1)
        outputs = torch.cat((outputs, right_most_feat), dim=-1)
        # else:
        #     raise Exception("not suppurted padding params")

        return outputs

    def _get_sampling_shift(self):
        # [-k//2, -k//2+1, ..., k//2]
        positions = torch.Tensor(torch.arange(-self.half_span, self.half_span+1, self.dilation).float())
        assert len(positions) == self.kernel_size
        return positions

    def _normalize_offset(self, idx, H):
        # normalize the idx to [-1, 1]
        return idx * 2.0 / (H - 1)

    def _make_grid(self, H, W, dH = 0., batch_size=0):
        '''
        dH -  offset relative to the ordinate idx, [b, h, w], elem value in [-1, 1]
        make grid for linear interpolation along H dimension
        [-1, -1] for left upper corner in H x W
        [1,  -1] for right upper corner in H x W
        [-1,  1] for left bottom corner in H x W
        [1,   1] for right bottom corner in H x W
        grid[b, h, w, 2] -> [y, x] -> [w_idx, h_idx]
        '''
        if isinstance(dH, float) or isinstance(dH, int): 
            # scaler shift, same shift along the H dimension
            h_grid = torch.arange(H).unsqueeze(1).repeat(1, W).unsqueeze(-1).float() 
            h_grid = h_grid.unsqueeze(0) / (H - 1) * 2.0 - 1.0 + dH
            w_grid = torch.arange(W).unsqueeze(0).repeat(H, 1).unsqueeze(-1).float()
            w_grid = w_grid.unsqueeze(0) / (W - 1) * 2.0 - 1.0

            if batch_size > 0:
                h_grid = h_grid.repeat(batch_size, 1, 1, 1)
                w_grid = w_grid.repeat(batch_size, 1, 1, 1)

        elif isinstance(dH, torch.Tensor) and len(dH.size()) == 3: # [b, h, w]
            # [h, w, 1]
            h_grid = torch.arange(H).unsqueeze(1).repeat(1, W).unsqueeze(-1).float()
            h_grid = h_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) / (H - 1) * 2.0 - 1.0
            h_grid += dH.unsqueeze(-1)
            h_grid = torch.clamp(h_grid, -1.0, 1.0)

            w_grid = torch.arange(W).unsqueeze(0).repeat(H, 1).unsqueeze(-1).float()
            w_grid = w_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) / (W - 1) * 2.0 - 1.0
        else:
            raise NotImplementedError("offsets size not supported")

        grid = torch.cat((w_grid, h_grid), dim=-1).cuda()
        return grid 

class TemporalDeformableLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding_mode: str, # 'border'
        offset_act: str,
        min_idx: int,
        max_idx: int,
        clamp_idx: bool,
        dropout: float,
        use_norm: bool 
        ):

        super(TemporalDeformableLayer, self).__init__()

        self.conv =  TemporalDeformableBlock(
                                    in_channels,
                                    out_channels * 2,
                                    kernel_size,
                                    stride,
                                    padding_mode,
                                    offset_act,
                                    min_idx,
                                    max_idx,
                                    clamp_idx,
                                    use_norm
                                )
        self.act = nn.GLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    # inputs : [batch, time_step, dim]
    # outputs : [batch, time_step, dim]
    def forward(self, inputs):
        cur_inputs = inputs.permute(0, 2, 1) # -> [batch, dim, time_step]
        outputs =  self.act(self.conv(cur_inputs)) # [b, time_step ,hidden]
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        return outputs

class ShiftedConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list, # list of int 
        stride: int,
        padding_mode: str, # 'zeros'
        dropout: float,
        use_norm: bool 
        ):

        super(ShiftedConvLayer, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = 1

        if use_norm:
            self.conv =  weight_norm(nn.Conv1d(
                                    in_channels,
                                    out_channels * 2,
                                    kernel_size,
                                    stride,
                                    self.kernel_size-1,
                                    padding_mode=padding_mode
                                )).cuda()   
        else:
            self.conv =  nn.Conv1d(
                                    in_channels,
                                    out_channels * 2,
                                    kernel_size,
                                    stride,
                                    self.kernel_size-1,
                                    padding_mode=padding_mode
                                ).cuda()
        self.act = nn.GLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None 
        
    # inputs : [b, time_step, dim]
    def forward(self, inputs):
        cur_inputs = inputs.permute(0, 2, 1) # [batch, dim, time_step]
        conv_outputs = self.conv(cur_inputs)
        if self.dropout is not None:
            conv_outputs = self.dropout(conv_outputs)
        conv_outputs = conv_outputs.permute(0, 2, 1)[:, :-self.kernel_size+1, :] # masked conv outputs, safe indexing
        outputs = self.act(conv_outputs)
        return outputs

class SoftAttention(nn.Module):
    def __init__(
        self, 
        *,
        hidden_size: int, 
        att_embed_size: int,
        att_embed_dropout: float,
        use_norm: bool
    ):
        super(SoftAttention, self).__init__()
        if use_norm:
            self.w_h = weight_norm(nn.Linear(hidden_size, att_embed_size, bias=False))
            self.w_alpha = weight_norm(nn.Linear(att_embed_size, 1, bias=False))
        else:
            self.w_h = nn.Linear(hidden_size, att_embed_size, bias=False)
            self.w_alpha = nn.Linear(att_embed_size, 1, bias=False)

        self.act = nn.Tanh()
        self.dropout = nn.Dropout(att_embed_dropout) if att_embed_dropout > 0 else None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states, att_feats, p_att_feats, att_masks = None, **kwargs):
        w_h = self.w_h(hidden_states).unsqueeze(1)
        alpha = self.act(w_h + p_att_feats)
        if (self.dropout is not None) and self.training:
            alpha = self.dropout(alpha)
        alpha = self.w_alpha(alpha).squeeze(-1)
        if att_masks is not None:
            alpha = alpha + att_masks
        alpha = self.softmax(alpha)
        att = torch.bmm(alpha.unsqueeze(1), att_feats).squeeze(1)
        return att
