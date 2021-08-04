"""
From original at https://github.com/aimagelab/meshed-memory-transformer/blob/master/models/transformer/utils.py
Original copyright of AImageLab code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) 2019, AImageLab
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PositionWiseFeedForward"]

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(
        self, 
        *,
        d_model: int, 
        d_ff: int, 
        dropout: float
    ):
        super(PositionWiseFeedForward, self).__init__()

        #self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0. else None
        self.dropout_2 = nn.Dropout(p=dropout) if dropout > 0. else None 
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        # if self.identity_map_reordering:
        #    out = self.layer_norm(input)
        #    out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
        #    out = input + self.dropout(torch.relu(out))
        #else:
        out = F.relu(self.fc1(inputs))
        if self.dropout_2:
            out = self.dropout_2(out)
        out = self.fc2(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.layer_norm(inputs + out)
        return out
