# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_predictor, build_v_predictor, add_predictor_config
from .base_predictor import BasePredictor
from .bert_predictor import BertPredictionHead, BertVisualPredictionHead
from .multimodal_predictor import MultiModalPredictor

__all__ = list(globals().keys())