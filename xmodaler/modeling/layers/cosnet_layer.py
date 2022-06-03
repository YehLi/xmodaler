import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from ..layers.create_act import get_activation
from .bert import BertAttention, BertIntermediate, BertOutput

class COSJointAttention(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        layer_norm_eps,
        hidden_dropout_prob
    ):
        super(COSJointAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.resid_dropout = nn.Dropout(hidden_dropout_prob)

        ##################################################################
        self.v_attn = nn.Linear(hidden_size, hidden_size * 2)
        self.o_attn = nn.Linear(hidden_size, hidden_size * 2)
        self.vo_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.gate_attn = nn.Linear(2 * hidden_size, hidden_size)
        self.gate = nn.Sigmoid()
        ##################################################################

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "num_attention_heads": cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
            "attention_probs_dropout_prob": cfg.MODEL.BERT.ATTENTION_PROBS_DROPOUT_PROB,
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB
        }

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)

        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)

    def attn(self, mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask):
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        outputs = context_layer.view(*new_context_layer_shape)
        return outputs

    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        v_feats, 
        o_feats,
        v_attention_mask, 
        o_attention_mask, 
        history_states=None    
    ):
        input_tensor = hidden_states

        mixed_query_layer = self.query(hidden_states)
        if history_states is not None:            
            mixed_key_layer = self.key(history_states)
            mixed_value_layer = self.value(history_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        v_mixed_key_layer, v_mixed_value_layer = self.v_attn(v_feats).split(self.all_head_size, dim=2)
        o_mixed_key_layer, o_mixed_value_layer = self.o_attn(o_feats).split(self.all_head_size, dim=2)

        l_outputs = self.attn(mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask)
        l_outputs = self.c_proj(l_outputs)

        v_outputs = self.attn(mixed_query_layer, v_mixed_key_layer, v_mixed_value_layer, v_attention_mask)
        o_outputs = self.attn(mixed_query_layer, o_mixed_key_layer, o_mixed_value_layer, o_attention_mask)
        vo_outputs = self.vo_proj(torch.cat([v_outputs, o_outputs], dim=-1))

        gate = self.gate_attn(torch.cat([l_outputs, vo_outputs], dim=-1))
        gate = self.gate(gate)
        outputs = gate * l_outputs + (1 - gate) * vo_outputs
        outputs = self.resid_dropout(outputs)
        outputs = self.LayerNorm(outputs + input_tensor)
        return outputs

class COSBertIntermediate(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size: int,
        hidden_act: str,
        intermediate_size: int,
        intermediate_drop: float
    ):
        super(COSBertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = get_activation(hidden_act)
        self.dropout = nn.Dropout(intermediate_drop)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "hidden_act": cfg.MODEL.BERT.HIDDEN_ACT,
            "intermediate_size": cfg.MODEL.BERT.INTERMEDIATE_SIZE,
            "intermediate_drop": cfg.MODEL.BERT.INTERMEDIATE_DROP
        }

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class COSBertOutput(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        ffn_dropout_prob: float
    ):
        super(COSBertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(ffn_dropout_prob)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "intermediate_size": cfg.MODEL.BERT.INTERMEDIATE_SIZE,
            "layer_norm_eps": 1e-12,
            "ffn_dropout_prob": cfg.MODEL.BERT.FFN_DROPOUT_PROB
        }

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states        

class COSNetDecBlock(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        attention,
        bert_intermediate,
        bert_output
    ):
        super(COSNetDecBlock, self).__init__()
        self.attn = attention
        self.intermediate = bert_intermediate
        self.output = bert_output

    @classmethod
    def from_config(cls, cfg):
        return {
            'attention': COSJointAttention(cfg),
            "bert_intermediate": COSBertIntermediate(cfg),
            "bert_output": COSBertOutput(cfg)
        }

    def forward(self, 
        lang_feats, 
        v_feats, 
        o_feats,
        lang_attention_mask=None, 
        v_attention_mask=None, 
        o_attention_mask=None,
        t_history_states=None
    ):
        x = self.attn(
            lang_feats, 
            lang_attention_mask, 
            v_feats, 
            o_feats,
            v_attention_mask, 
            o_attention_mask,
            t_history_states
        )
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)
        return layer_output