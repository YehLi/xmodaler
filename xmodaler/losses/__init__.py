# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_losses, build_rl_losses, add_loss_config

from .cross_entropy import CrossEntropy
from .label_smoothing import LabelSmoothing
from .pretrain_losses import PretrainLosses
from .reward_criterion import RewardCriterion
from .bce_logits import BCEWithLogits
from .triplet import BatchTriplet, Triplet
from .semcomphder_loss import SemComphderLoss