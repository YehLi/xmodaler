# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import load_vocab, decode_sequence, decode_sequence_bert
from xmodaler.tokenization import BertTokenizer

class DecodeStrategy(nn.Module, metaclass=ABCMeta):
    @configurable
    def __init__(
        self,
        *,
        vocab_path,
        beam_size,
        max_seq_len,
        bert_tokenizer,
        bos_token_id,
        eos_token_id
    ):
        super().__init__()
        self.beam_size = beam_size
        if bert_tokenizer is None:
            self.vocab = load_vocab(vocab_path)
        else:
            self.vocab = None

        self.max_seq_len = max_seq_len
        self.bert_tokenizer = bert_tokenizer
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @classmethod
    def from_config(cls, cfg):
        tokenizer_map = {
            'BERT': BertTokenizer
        }

        tokenizer_cls = tokenizer_map.get(cfg.INFERENCE.VOCAB, None)
        if tokenizer_cls is None:
            bert_tokenizer = None
            bos_token_id = 0
            eos_token_id = 0
        else:
            bert_tokenizer = tokenizer_cls.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME, do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE)
            if cfg.INFERENCE.VOCAB == 'BERT':
                bos_token_id = bert_tokenizer.vocab["[CLS]"]
                eos_token_id = bert_tokenizer.vocab["[SEP]"]

        return {
            "vocab_path": cfg.INFERENCE.VOCAB,
            "beam_size": cfg.DECODE_STRATEGY.BEAM_SIZE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            'bert_tokenizer': bert_tokenizer,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id
        }

    @abstractmethod
    def _forward(self, batched_inputs, model):
        pass

    def forward(self, batched_inputs, output_sents, model):
        ret = self._forward(batched_inputs, model)
        if output_sents:
            if self.vocab:
                sents = decode_sequence(self.vocab, ret[kfg.G_SENTS_IDS])
            else:
                sents = decode_sequence_bert(self.bert_tokenizer, ret[kfg.G_SENTS_IDS], self.eos_token_id)
            ret.update({ kfg.OUTPUT: sents })
        return ret