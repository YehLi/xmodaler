# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import time
import torch
import numpy as np
from .defaults import DefaultTrainer
from xmodaler.scorer import build_scorer
from xmodaler.config import kfg
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY

__all__ = ['RLBeamTrainer']

@ENGINE_REGISTRY.register()
class RLBeamTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(RLBeamTrainer, self).__init__(cfg)
        self.scorer = self.build_scorer(cfg)

    @classmethod
    def build_scorer(cls, cfg):
        return build_scorer(cfg)

    def run_step(self):
        start = time.perf_counter()
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)
        data_time = time.perf_counter() - start

        data = comm.unwrap_model(self.model).preprocess_batch(data)

        beam_size = comm.unwrap_model(self.model).beam_searcher.beam_size
        batch_size = data['ATT_FEATS'].shape[0]
        device = data['ATT_FEATS'].device

        self.model.train()
        # set out_size in beam search decode
        data['OUT_SIZE'] = beam_size
        outputs_dict = self.model(data, use_beam_search=True, output_sents=False)
        
        # repeat IDS to compute cider rewards
        outputs_dict[kfg.IDS] = np.repeat(np.expand_dims(np.array(outputs_dict[kfg.IDS]).flatten(), axis=1), beam_size, axis=1).flatten().tolist()
        outputs_dict[kfg.G_SENTS_IDS] = outputs_dict[kfg.G_SENTS_IDS].view(batch_size*beam_size, -1)
        rewards = self.scorer(outputs_dict)

        reward = rewards[kfg.REWARDS].astype(np.float32)
        reward = torch.from_numpy(reward).to(device).view(batch_size, beam_size)
        reward_baseline = torch.mean(reward, -1, keepdim=True)

        loss = -torch.mean(outputs_dict[kfg.G_LOGP], -1) * (reward - reward_baseline)
        loss = loss.mean()
        
        losses_dict = { 'BeamRewardCriterion': loss }
        losses = sum(losses_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        rewards.pop(kfg.REWARDS)
        losses_dict.update(rewards)
        self._write_metrics(losses_dict, data_time)
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)