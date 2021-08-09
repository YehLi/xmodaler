# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
from xmodaler.functional import load_vocab
from xmodaler.config import configurable
from .build import SCORER_REGISTRY
from .ptb_tokenizer import PTBTokenizer
from .base_scorer import BaseScorer

__all__ = ['PTBTokenizedScorer']

@SCORER_REGISTRY.register()
class PTBTokenizedScorer(BaseScorer):
    @configurable
    def __init__(
        self,
        *,
        types,
        scorers,
        weights,
        gt_path,
        eos_id,
        vocab_path
    ): 
        super(PTBTokenizedScorer, self).__init__(
            types = types,
            scorers = scorers,
            weights = weights,
            gt_path = gt_path,
            eos_id = eos_id
        )
        self.vocab = load_vocab(vocab_path)

        # update train gts raw with PTBTokenizer preprocessing
        train_raw_sents = self.gts
        idx2imgid = {i:imgid for i,imgid in enumerate(list(train_raw_sents.keys()))}
        cap_gt = PTBTokenizer.tokenize(list(train_raw_sents.values()))
        self.gts = {idx2imgid[i]:[s.split() for s in sents] for i, sents in cap_gt.items()}

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({ 'vocab_path': cfg.INFERENCE.VOCAB })
        return ret

    def get_sents(self, sent):
        words = []
        for ix in sent:
            if ix == self.eos_id:
                # words.append('.')
                break
            words.append(self.vocab[ix])
        return words