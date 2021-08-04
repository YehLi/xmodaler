# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import numpy as np
import pickle

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.tokenization import BertTokenizer
from .build import SCORER_REGISTRY

__all__ = ['BertTokenizedScorer']

@SCORER_REGISTRY.register()
class BertTokenizedScorer(object):
    @configurable
    def __init__(
        self,
        *,
        types,
        scorers,
        weights,
        gt_path,
        bert_tokenizer
    ): 
       self.types = types
       self.scorers = scorers
       self.weights = weights
       self.gts = pickle.load(open(gt_path, 'rb'), encoding='bytes')
       self.gts = {str(k):v for k,v in self.gts.items()}

       self.tokenizer = bert_tokenizer
       self.sep_token_id = self.tokenizer.vocab["[SEP]"]

    @classmethod
    def from_config(cls, cfg):
        scorers = []
        for name in cfg.SCORER.TYPES:
            scorers.append(SCORER_REGISTRY.get(name)(cfg))

        return {
            'scorers': scorers,
            'types': cfg.SCORER.TYPES,
            'weights': cfg.SCORER.WEIGHTS,
            'gt_path': cfg.SCORER.GT_PATH,
            'bert_tokenizer': BertTokenizer.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME,
                do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE)
        }

    def get_sents(self, sent):
        words = []
        for word in sent:
            if word == self.sep_token_id:
                words.append('.')
                break
            words.append(self.tokenizer.ids_to_tokens[word])
            
        words = self.tokenizer.convert_tokens_to_string(words).split()
        return words

    def __call__(self, batched_inputs):
        ids = batched_inputs[kfg.IDS]
        res = batched_inputs[kfg.G_SENTS_IDS]
        res = res.cpu().tolist()

        hypo = [self.get_sents(r) for r in res]
        gts = [self.gts[i] for i in ids]

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, hypo)
            rewards += self.weights[i] * scores
            rewards_info[self.types[i]] = score
        rewards_info.update({ kfg.REWARDS: rewards })
        return rewards_info

