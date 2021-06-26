import torch
from torch import nn

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