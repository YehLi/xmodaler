# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import time
import copy
import tqdm
import logging
import numpy as np
import weakref

import torch
from .defaults import DefaultTrainer
from xmodaler.scorer import build_scorer
from xmodaler.config import kfg
from xmodaler.losses import build_rl_losses
import xmodaler.utils.comm as comm
from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.modeling.meta_arch.ensemble import Ensemble
from . import hooks
from .train_loop import TrainerBase
from .build import ENGINE_REGISTRY

__all__ = ['Ensembler']

@ENGINE_REGISTRY.register()
class Ensembler(DefaultTrainer):
    def __init__(self, cfg):
        super(Ensembler, self).__init__(cfg)
        models = []
        num_models = len(cfg.MODEL.ENSEMBLE_WEIGHTS)
        assert num_models > 0, "cfg.MODEL.ENSEMBLE_WEIGHTS is empty"
        for i in range(num_models):
            models.append(copy.deepcopy(self.model))
            
            checkpointer = XmodalerCheckpointer(
                models[i],
                cfg.OUTPUT_DIR,
                trainer=weakref.proxy(self),
            )
            checkpointer.resume_or_load(cfg.MODEL.ENSEMBLE_WEIGHTS[i], resume=False)

        self.model = Ensemble(models, cfg)
        self.ema = None

    def resume_or_load(self, resume=True):
        pass
            
    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        model.eval()
        results = []
        with torch.no_grad():
            for data in tqdm.tqdm(test_data_loader):
                data = comm.unwrap_model(model).preprocess_batch(data)
                ids = data[kfg.IDS]

                if cfg.INFERENCE.GENERATION_MODE == True:
                    res = model(data, use_beam_search=True, output_sents=True)
                else:
                    res = model(data)

                outputs = res[kfg.OUTPUT]
                for id, output in zip(ids, outputs):
                    results.append({cfg.INFERENCE.ID_KEY: int(id), cfg.INFERENCE.VALUE: output})

        if evaluator is not None:
            eval_res = evaluator.eval(results, epoch)
        else:
            eval_res = ''
        return eval_res