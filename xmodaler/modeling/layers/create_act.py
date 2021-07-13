"""
From original at https://github.com/huggingface/transformers and https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/create_act.py
Modifications by Yehao Li, Copyright 2021.	
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["get_act_layer", "get_activation"]

########################################### Layer ###########################################
_ACT_LAYER_DEFAULT = dict(
    relu=nn.ReLU,
    elu=nn.ELU,
    celu=nn.CELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
)

def get_act_layer(name='none'):
    if name in _ACT_LAYER_DEFAULT:
        return _ACT_LAYER_DEFAULT[name]
    else:
        return None

########################################### Function ###########################################
def swish(x):
    return x * torch.sigmoid(x)

def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

gelu = getattr(F, "gelu", _gelu_python)

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": F.tanh,
    "gelu_new": gelu_new,
    "mish": mish
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            "function {} not found in ACT2FN mapping {} or torch.nn.functional".format(
                activation_string, list(ACT2FN.keys())
            )
        )
