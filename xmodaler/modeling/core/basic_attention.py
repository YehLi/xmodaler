import torch
import torch.nn as nn

__all__ = ["BasicAttention"]

class BasicAttention(nn.Module):
    def __init__(
        self, 
        *,
        hidden_size: int, 
        att_embed_size: int,
        att_embed_dropout: float
    ):
        super(BasicAttention, self).__init__()
        self.w_h = nn.Linear(hidden_size, att_embed_size, bias=False)
        self.act = nn.Tanh()
        self.w_alpha = nn.Linear(att_embed_size, 1, bias=False)
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