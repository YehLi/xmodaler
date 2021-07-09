import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..layers.bert import BertPredictionHeadTransform
from .build import PREDICTOR_REGISTRY

__all__ = ["BertPredictionHead", "BertVisualPredictionHead"]

@PREDICTOR_REGISTRY.register()
class BertPredictionHead(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size,
        vocab_size,
        transform
    ):
        super(BertPredictionHead, self).__init__()
        self.transform = transform

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "transform": BertPredictionHeadTransform(cfg)
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ret = {}
        if kfg.U_HIDDEN_STATES in batched_inputs:
           hidden_states = batched_inputs[kfg.U_HIDDEN_STATES]
           if isinstance(hidden_states, list):
               hidden_states = hidden_states[-1]

           hidden_states = self.transform(hidden_states)
           u_logits = self.decoder(hidden_states)
           ret.update({ kfg.U_LOGITS: u_logits })

        if kfg.G_HIDDEN_STATES in batched_inputs:
           hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
           if isinstance(hidden_states, list):
               hidden_states = hidden_states[-1]

           hidden_states = self.transform(hidden_states)
           g_logits = self.decoder(hidden_states)
           ret.update({ kfg.G_LOGITS: g_logits })
        return ret

@PREDICTOR_REGISTRY.register()
class BertVisualPredictionHead(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size,
        v_target_size,
        transform
    ):
        super(BertVisualPredictionHead, self).__init__()
        self.transform = transform

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, v_target_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(v_target_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "v_target_size": cfg.MODEL.BERT.V_TARGET_SIZE,
            "transform": BertPredictionHeadTransform(cfg)
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        hidden_states = batched_inputs[kfg.ATT_FEATS]
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]

        hidden_states = self.transform(hidden_states)
        logits = self.decoder(hidden_states)
        return { kfg.V_LOGITS: logits }