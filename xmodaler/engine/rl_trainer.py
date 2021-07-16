# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import time
import copy
import torch
from .defaults import DefaultTrainer
from xmodaler.scorer import build_scorer
from xmodaler.config import kfg
from xmodaler.losses import build_rl_losses
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY

__all__ = ['RLTrainer']

@ENGINE_REGISTRY.register()
class RLTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(RLTrainer, self).__init__(cfg)
        self.scorer = self.build_scorer(cfg)
        self.losses = build_rl_losses(cfg)

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

        self.model.eval()
        with torch.no_grad():
            bs_data = copy.copy(data)
            bs_outputs_dict = self.model(bs_data, use_beam_search=False, output_sents=False)
        bs_rewards = self.scorer(bs_outputs_dict)

        self.model.train()
        data[kfg.DECODE_BY_SAMPLE] = True
        outputs_dict = self.model(data, use_beam_search=False, output_sents=False)
        rewards = self.scorer(outputs_dict)
        rewards = torch.from_numpy(rewards[kfg.REWARDS] - bs_rewards[kfg.REWARDS]).float().cuda()
        outputs_dict.update({ kfg.REWARDS: rewards })

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
        losses = sum(losses_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        bs_rewards.pop(kfg.REWARDS)
        loss_dict.update(bs_rewards)
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()