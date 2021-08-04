# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/checkpoint/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# File:

from .xmodaler_checkpoint import XmodalerCheckpointer, PeriodicEpochCheckpointer
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer