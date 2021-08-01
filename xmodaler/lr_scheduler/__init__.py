# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_lr_scheduler

from .step_lr import StepLR
from .noam_lr import NoamLR
from .warmup_lr import (
   WarmupConstant, 
   WarmupLinear, 
   WarmupCosine, 
   WarmupCosineWithHardRestarts, 
   WarmupMultiStepLR
)
from .multi_step_lr import MultiStepLR
from .fix_lr import FixLR