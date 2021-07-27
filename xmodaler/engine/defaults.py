# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""
import time
import argparse
import logging
import tqdm
import os
import sys
import numpy as np
import weakref
from collections import OrderedDict
from typing import Dict, List, Optional
from omegaconf import OmegaConf

import torch
from torch.nn.parallel import DistributedDataParallel

from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.datasets import build_xmodaler_train_loader, build_xmodaler_valtest_loader
from xmodaler.modeling import build_model
from xmodaler.optim import build_optimizer
from xmodaler.lr_scheduler import build_lr_scheduler
from xmodaler.evaluation import build_evaluation
from xmodaler.losses import build_losses
from xmodaler.config import kfg
from xmodaler.utils import comm
from xmodaler.utils.collect_env import collect_env_info
from xmodaler.utils.env import TORCH_VERSION, seed_all_rng
from xmodaler.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, get_event_storage
from xmodaler.utils.file_io import PathManager
from xmodaler.utils.logger import setup_logger

from . import hooks
from .train_loop import TrainerBase
from .build import ENGINE_REGISTRY

__all__ = [
    "default_argument_parser",
    "default_setup",
    "default_writers",
    "DefaultTrainer",
]

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by X-modaler users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser    

def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the X-modaler logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]

@ENGINE_REGISTRY.register()
class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`DefaultTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`DefaultTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`DefaultTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in X-modaler.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (XmodalerCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("xmodaler")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        #cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, model)
        self.train_data_loader = self.build_train_loader(cfg)
        self.iters_per_epoch = len(self.train_data_loader)
        self._train_data_loader_iter = iter(self.train_data_loader)
        self.val_data_loader = self.build_val_loader(cfg)
        self.test_data_loader = self.build_test_loader(cfg)
        
        if self.val_data_loader is not None:
            self.val_evaluator = build_evaluation(cfg, cfg.INFERENCE.VAL_ANNFILE, None)
        else:
            self.val_evaluator = None

        if self.test_data_loader is not None:
            self.test_evaluator = build_evaluation(cfg, cfg.INFERENCE.TEST_ANNFILE, cfg.OUTPUT_DIR)
        else:
            self.test_evaluator = None

        self.losses = self.build_losses(cfg)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, self.iters_per_epoch)
        self.ss_prob = 0.0
        
        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, find_unused_parameters=True, 
                device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        self.model = model
        self.model.train()

        self.checkpointer = XmodalerCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self.cfg = cfg
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.EPOCH * self.iters_per_epoch
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.ScheduledSampling(
                start_iter = cfg.SCHEDULED_SAMPLING.START_EPOCH * self.iters_per_epoch, 
                inc_every_iter = cfg.SCHEDULED_SAMPLING.INC_EVERY_EPOCH * self.iters_per_epoch, 
                inc_prob = cfg.SCHEDULED_SAMPLING.INC_PROB, 
                max_prob = cfg.SCHEDULED_SAMPLING.MAX_PROB
            )
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD * self.iters_per_epoch))

        def test_and_save_results(epoch):
            eval_results = self.test(self.cfg, self.model, self.test_data_loader, self.test_evaluator, epoch)
            return eval_results

        def val_and_save_results(epoch):
            eval_results = self.test(self.cfg, self.model, self.val_data_loader, self.val_evaluator, epoch)
            return eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        if self.val_data_loader is not None:
            ret.append(
                hooks.EvalHook(
                    eval_period = cfg.SOLVER.EVAL_PERIOD, 
                    eval_start = cfg.INFERENCE.VAL_EVAL_START,
                    eval_function = val_and_save_results, 
                    iters_per_epoch = self.iters_per_epoch,
                    stage = 'val'
                ))

        if self.test_data_loader is not None:
            ret.append(
                hooks.EvalHook(
                    eval_period = cfg.SOLVER.EVAL_PERIOD, 
                    eval_start = cfg.INFERENCE.TEST_EVAL_START,
                    eval_function = test_and_save_results, 
                    iters_per_epoch = self.iters_per_epoch,
                    stage = 'test'
                ))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.WRITE_PERIOD))
        return ret

    def build_writers(self):
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        super().train(self.start_iter, self.max_iter)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, iters_per_epoch):
        return build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_xmodaler_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        return build_xmodaler_valtest_loader(cfg, stage='test')

    @classmethod
    def build_val_loader(cls, cfg):
        return build_xmodaler_valtest_loader(cfg, stage='val')

    @classmethod
    def build_losses(cls, cfg):
        return build_losses(cfg)

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        pass

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics_dict.update({ k: v.detach().cpu().item() })
            else:
                metrics_dict.update({ k: v })
        #metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

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
        model.train()
        return eval_res

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
        data[kfg.SS_PROB] = self.ss_prob
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

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()