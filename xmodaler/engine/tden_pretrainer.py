# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import time
import tqdm
import copy
import numpy as np
import itertools
import torch
from .defaults import DefaultTrainer
from xmodaler.scorer import build_scorer
from xmodaler.config import kfg
from xmodaler.losses import build_rl_losses
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY

__all__ = ['TDENPretrainer']

@ENGINE_REGISTRY.register()
class TDENPretrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(TDENPretrainer, self).__init__(cfg)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        data = comm.unwrap_model(self.model).preprocess_batch(data)
        data_copy = copy.deepcopy(data)
        data[kfg.SS_PROB] = self.ss_prob
        data['tden_pretrain_similarity'] = True
        outputs_dict = self.model(data)

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
        losses = sum(losses_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(losses_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

        u_tokens_ids = data[kfg.U_TOKENS_IDS]
        u_tlogits = outputs_dict[kfg.U_LOGITS]
        u_targets = data[kfg.U_TARGET_IDS]
        u_mask = (u_targets < 0).type(torch.cuda.IntTensor)
        _, wt_u = torch.max(u_tlogits.detach(), -1)
        new_u_tokens_ids = u_mask * u_tokens_ids + (1 - u_mask) * wt_u
        data_copy[kfg.U_TOKENS_IDS] = new_u_tokens_ids

        g_tlogits = outputs_dict[kfg.G_LOGITS]
        _, wt_g = torch.max(g_tlogits.detach(), -1)
        new_g_tokens_ids = u_mask * u_tokens_ids + (1 - u_mask) * torch.cat([wt_g.new(wt_g.size(0), 1), wt_g[:, 0:-1]], dim=-1)
        data_copy[kfg.G_TOKENS_IDS] = new_g_tokens_ids
        data = data_copy

        # second stage
        data['tden_pretrain_similarity'] = False
        outputs_dict = self.model(data)

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
        losses = sum(losses_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(losses_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()


    