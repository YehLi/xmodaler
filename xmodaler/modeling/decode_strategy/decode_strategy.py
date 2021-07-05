from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import load_vocab, decode_sequence

class DecodeStrategy(nn.Module, metaclass=ABCMeta):
    @configurable
    def __init__(
        self,
        *,
        vocab_path,
        beam_size,
        max_seq_len
    ):
        super().__init__()
        self.beam_size = beam_size
        self.vocab = load_vocab(vocab_path)
        self.max_seq_len = max_seq_len

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_path": cfg.INFERENCE.VOCAB,
            "beam_size": cfg.DECODE_STRATEGY.BEAM_SIZE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN
        }

    @abstractmethod
    def _forward(self, batched_inputs, model):
        pass

    def forward(self, batched_inputs, output_sents, model):
        ret = self._forward(batched_inputs, model)
        if output_sents:
            sents = decode_sequence(self.vocab, ret[kfg.G_SENTS_IDS])
            ret.update({ kfg.G_SENTS: sents })
        return ret