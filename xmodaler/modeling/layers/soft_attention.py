import torch
import torch.nn as nn

__all__ = ["SoftAttention"]

class SoftAttention(nn.Module):
    def __init__(
        self, 
        *,
        hidden_size: int, 
        att_feat_size: int,
        att_embed_size: int,
        att_embed_dropout: float
    ):
        super(SoftAttention, self).__init__()

        self.w_h = nn.Linear(hidden_size, att_embed_size, bias=False)
        self.w_u = nn.Linear(att_feat_size, att_embed_size, bias=False)
        self.b = nn.Parameter(torch.ones(att_embed_size), requires_grad=True)
        self.act = nn.Tanh()
        self.w = nn.Linear(att_embed_size, 1, bias=False)

        self.dropout = nn.Dropout(att_embed_dropout) if att_embed_dropout > 0 else None
        self.softmax = nn.Softmax(dim=-1)

    '''
    hidden : supported size [batch, input_size] or [batch, seq_len, input_size]
    '''
    def forward(self, hidden, feats, att_mask=None, **kwargs):
        n_dim = hidden.dim()
        if n_dim == 3:
            batch_size, seq_len, hidden_dim = hidden.size()
            hidden = hidden.view(-1, hidden_dim)

            _, num_att, att_feat_dim = feats.size()
            feats = feats.unsqueeze(1).repeat(1, seq_len, 1, 1).view(-1, num_att, att_feat_dim)
            att_mask = att_mask.unsqueeze(1).repeat(1, seq_len, 1).view(-1, num_att)

        Wh = self.w_h(hidden) # Wa * h(t-1)
    
        Uv = self.w_u(feats) # Ua * vi [batch_size, num_att, dim]

        Wh = Wh.unsqueeze(1).expand_as(Uv)

        attn_weights = self.act(Wh + Uv + self.b)

        if self.dropout:
            attn_weights = self.dropout(attn_weights)

        attn_weights = self.w(attn_weights) # [batch_size * seq_len, num_att, 1]

        if att_mask is not None:
            attn_weights = attn_weights.masked_fill(att_mask.unsqueeze(-1) == 0, -1e9)

        weights = self.softmax(attn_weights.squeeze(2)).unsqueeze(2)

        weighted_feats = feats * weights.expand_as(feats) # element-wise
        
        attn_feats = weighted_feats.sum(dim=1) # weighted sum

        if n_dim == 3:
            attn_dim = attn_feats.size(-1)
            attn_feats = attn_feats.view(batch_size, -1, attn_dim)
            weights = weights.view(batch_size, seq_len, -1)
        
        return attn_feats, weights