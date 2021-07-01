import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import PREDICTOR_REGISTRY

__all__ = ["SALSTMPredictor"]

@PREDICTOR_REGISTRY.register()
class SALSTMPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        pred_ctx_size: int,
        hidden_size: int, 
        ctx_dropout: float,
        lm_dropout: float,
        vocab_size: int   # include <BOS>/<EOS>
    ):
        super(SALSTMPredictor, self).__init__()

        self.ctx_drop = nn.Dropout(ctx_dropout) if ctx_dropout > 0 else None
        self.out_proj = nn.Linear(pred_ctx_size, hidden_size)
        self.tanh = nn.Tanh()
        self.lm_drop = nn.Dropout(lm_dropout) if lm_dropout > 0 else None
        self.logits = nn.Linear(hidden_size, vocab_size)
        #self.vocab_size = vocab_size
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "pred_ctx_size": cfg.MODEL.DECODER_DIM + cfg.MODEL.TOKEN_EMBED.DIM + cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "ctx_dropout": cfg.MODEL.SALSTM.CTX_DROPOUT,
            "lm_dropout": cfg.MODEL.SALSTM.LM_DROPOUT,
            "vocab_size": cfg.MODEL.VOCAB_SIZE
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ctx = batched_inputs[kfg.PREDICTOR_CTX]
        if self.ctx_drop:
            ctx = self.ctx_drop(ctx)
        ctx_embed = self.tanh(self.out_proj(ctx))
        
        if self.lm_drop:
            ctx_embed = self.lm_drop(ctx_embed)
            
        logits = self.logits(ctx_embed)
        return logits