# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_predictor, build_v_predictor, build_predictor_with_name, add_predictor_config
from .base_predictor import BasePredictor
from .bert_predictor import BertPredictionHead, BertVisualPredictionHead, BertVisualFeatureRegressionHead, BertIsMatchedPredictor
from .multimodal_predictor import MultiModalPredictor, SingleStreamMultiModalPredictor
from .multimodal_similarity import MultiModalSimilarity

__all__ = list(globals().keys())