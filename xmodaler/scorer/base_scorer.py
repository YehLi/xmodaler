# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import numpy as np
import pickle

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import SCORER_REGISTRY

__all__ = ['BaseScorer']

@SCORER_REGISTRY.register()
class BaseScorer(object):
    @configurable
    def __init__(
        self,
        *,
        types,
        scorers,
        weights,
        gt_path,
        eos_id
    ): 
       self.types = types
       self.scorers = scorers
       self.eos_id = eos_id
       self.weights = weights
       self.gts = pickle.load(open(gt_path, 'rb'), encoding='bytes')

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
            'eos_id': cfg.SCORER.EOS_ID
        }

    def get_sents(self, sent):
        words = []
        for word in sent:
            if word == self.eos_id:
                words.append(self.eos_id)
                break
            words.append(word)
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

