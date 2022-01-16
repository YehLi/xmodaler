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
from xmodaler.datasets import build_xmodaler_train_loader, build_xmodaler_valtest_loader, build_dataset_mapper
from .build import ENGINE_REGISTRY

__all__ = ['VCRTrainer']

@ENGINE_REGISTRY.register()
class VCRTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(VCRTrainer, self).__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        q2a_dataset_mapper = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train;VCR_Q-A")
        q2a_data_loader = build_xmodaler_train_loader(cfg, dataset_mapper=q2a_dataset_mapper)

        qa2r_dataset_mapper = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train;VCR_QA-R")
        qa2r_data_loader = build_xmodaler_train_loader(cfg, dataset_mapper=qa2r_dataset_mapper)
        return [q2a_data_loader, qa2r_data_loader]

    @classmethod
    def build_val_loader(cls, cfg):
        q2a_dataset_mapper = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="val;VCR_Q-A")
        q2a_data_loader = build_xmodaler_valtest_loader(cfg, dataset_mapper=q2a_dataset_mapper)

        qa2r_dataset_mapper = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="val;VCR_QA-R")
        qa2r_data_loader = build_xmodaler_valtest_loader(cfg, dataset_mapper=qa2r_dataset_mapper)
        return [q2a_data_loader, qa2r_data_loader]

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        model.eval()
        results_list = []
        with torch.no_grad():
            for i in range(len(test_data_loader)):
                results = []
                for data in tqdm.tqdm(test_data_loader[i]):
                    data = comm.unwrap_model(model).preprocess_batch(data)
                    res = model(data)

                    u_logits = res[kfg.U_LOGITS]
                    u_logits = u_logits.view(-1, cfg.DATALOADER.SEQ_PER_SAMPLE)
                    questions_ids = data[kfg.IDS].reshape((-1, cfg.DATALOADER.SEQ_PER_SAMPLE))[:, 0]

                    probs = torch.softmax(u_logits, dim=-1)
                    outputs = torch.max(probs, 1)[1].data.cpu().numpy()
                    targets = data[kfg.U_TARGET_IDS].view(-1).data.cpu().numpy()
                    for id, output, target in zip(questions_ids, outputs, targets):
                        results.append({ cfg.INFERENCE.ID_KEY: int(id), cfg.INFERENCE.VALUE: output, kfg.U_TARGET_IDS: target })
                results_list.append(results)

        if evaluator is not None:
            eval_res = evaluator.eval(results_list, epoch)
        else:
            eval_res = ''
        model.train()
        return eval_res

    def run_step(self):
        for i in range(len(self.train_data_loader)):
            assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
            start = time.perf_counter()
            
            try:
                data = next(self._train_data_loader_iter_list[i])
            except StopIteration:
                self._train_data_loader_iter_list[i] = iter(self.train_data_loader[i])
                data = next(self._train_data_loader_iter_list[i])

            data_time = time.perf_counter() - start

            data = comm.unwrap_model(self.model).preprocess_batch(data)
            data[kfg.SS_PROB] = self.ss_prob
            outputs_dict = self.model(data)

            u_logits = outputs_dict[kfg.U_LOGITS]
            u_logits = u_logits.view(-1, self.cfg.DATALOADER.SEQ_PER_SAMPLE)
            outputs_dict.update({ kfg.U_LOGITS: u_logits })

            losses_dict = {}
            for loss in self.losses:
                loss_dict = loss(outputs_dict)
                losses_dict.update(loss_dict)
            losses = sum(losses_dict.values())

            self.optimizer.zero_grad()
            losses.backward()

            self._write_metrics(losses_dict, data_time)
            self.optimizer.step()
        