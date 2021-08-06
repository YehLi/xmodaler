"""	
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/build.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import numpy as np
import operator
import pickle
from tabulate import tabulate
from termcolor import colored
import torch.utils.data
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from xmodaler.config import configurable
from xmodaler.utils.comm import get_world_size
from xmodaler.utils.env import seed_all_rng
from xmodaler.utils.file_io import PathManager
from xmodaler.utils.logger import log_first_n
from xmodaler.utils.registry import Registry
from .common import DatasetFromList, MapDataset

DATASETS_REGISTRY = Registry("DATASETS")  # noqa F401 isort:skip
DATASETS_REGISTRY.__doc__ = """
Registry for datasets, i.e. the whole model
"""

def build_dataset_mapper(cfg, name, stage):
    dataset_mapper = DATASETS_REGISTRY.get(name)(cfg, stage)
    return dataset_mapper

def trivial_batch_collator(batch):
    return batch

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)

def _train_loader_from_config(cfg, dataset_mapper=None, *, datalist=None, sampler=None):
    if len(cfg.DATASETS.TRAIN) > 0:
        if dataset_mapper is None:
            dataset_mapper = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train")
        if datalist is None:
            datalist = dataset_mapper.load_data(cfg)
    else:
        dataset_mapper = None
        datalist = None

    return {
        "datalist": datalist,
        "dataset_mapper": dataset_mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "batch_size": cfg.DATALOADER.TRAIN_BATCH_SIZE
    }

@configurable(from_config=_train_loader_from_config)
def build_xmodaler_train_loader(datalist, *, dataset_mapper, batch_size, num_workers):
    if datalist is None or dataset_mapper is None:
        return None

    if isinstance(datalist, list):
        dataset = DatasetFromList(datalist, copy=False)
    if dataset_mapper is not None:
        dataset = MapDataset(dataset, dataset_mapper)
    
    world_size = get_world_size()
    if world_size > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler = sampler,
        batch_size = batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        drop_last = True
    )
    return data_loader


def _valtest_loader_from_config(cfg, dataset_mapper=None, *, datalist=None, sampler=None, stage="val"):
    dataset_names = {
        "val": cfg.DATASETS.VAL,
        "test": cfg.DATASETS.TEST,
    }
    dataset_name = dataset_names[stage]
    if len(dataset_name) > 0:
        if dataset_mapper is None:
            dataset_mapper = build_dataset_mapper(cfg, name=dataset_name, stage=stage)
        if datalist is None:
            datalist = dataset_mapper.load_data(cfg)
    else:
        dataset_mapper = None
        datalist = None

    if dataset_name == 'Flickr30kDatasetForSingleStreamVal':
        multi_gpu_eval = True
        batch_size = 1
    else:
        multi_gpu_eval = False
        batch_size = cfg.DATALOADER.TEST_BATCH_SIZE

    return {
        "datalist": datalist,
        "dataset_mapper": dataset_mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "batch_size": batch_size,
        "multi_gpu_eval": multi_gpu_eval
    }

@configurable(from_config=_valtest_loader_from_config)
def build_xmodaler_valtest_loader(datalist, *, dataset_mapper, batch_size, num_workers, multi_gpu_eval):
    if datalist is None or dataset_mapper is None:
        return None

    if isinstance(datalist, list):
        dataset = DatasetFromList(datalist, copy=False)
    if dataset_mapper is not None:
        dataset = MapDataset(dataset, dataset_mapper)

    if multi_gpu_eval and get_world_size() > 1:
        # multi-gpu-eval for single stream retrieval
        sampler = DistributedSampler(dataset, shuffle=False)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            sampler=sampler,
            collate_fn=trivial_batch_collator,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            drop_last=False,
            shuffle = False,
            collate_fn=trivial_batch_collator,
        )
    return data_loader

