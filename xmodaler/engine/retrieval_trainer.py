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

__all__ = ['RetrievalTrainer']

@ENGINE_REGISTRY.register()
class RetrievalTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(RetrievalTrainer, self).__init__(cfg)

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        model.eval()

        ids = []
        vfeats = []
        tfeats = []
        with torch.no_grad():
            for data in tqdm.tqdm(test_data_loader):
                data = comm.unwrap_model(model).preprocess_batch(data)
                outputs = model(data)[kfg.OUTPUT]
                ids += data[kfg.IDS]
                vfeats.append(outputs[0])
                tfeats.append(outputs[1])

        iids = [ i[0] for i in ids]
        cids = [ i[1] for i in ids]
        cids = list(itertools.chain.from_iterable(cids))
        labels = np.expand_dims(cids, axis=1) == np.expand_dims(iids, axis=0)
        labels = labels.astype(int)
        vfeats = torch.cat(vfeats, dim=0)
        tfeats = torch.cat(tfeats, dim=0)

        if evaluator is not None:
            eval_res = evaluator.eval(vfeats, tfeats, labels)
        model.train()
        return eval_res