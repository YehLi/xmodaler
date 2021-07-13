# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn

__all__ = ["AttentionPooler"]

class AttentionPooler(nn.Module):
    def __init__(
        self, 
        *,
        hidden_size: int, 
        output_size: int,
        dropout: float,
        use_bn: bool
    ):
        super(AttentionPooler, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )
        self.embed = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm1d(output_size) if use_bn else None

    def forward(self, hidden_states, masks = None, **kwargs):
        score = self.att(hidden_states).squeeze(-1)
        if masks is not None:
            score = score + masks.view(score.size(0), -1)
        score = self.softmax(score)
        output = score.unsqueeze(1).matmul(hidden_states).squeeze(1)
        output = self.embed(output)
        if self.bn is not None:
            output = self.bn(output)
        return output