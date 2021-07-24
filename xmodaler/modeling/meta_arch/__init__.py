# -*- coding: utf-8 -*-
"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model, add_config
from .rnn_att_enc_dec import RnnAttEncoderDecoder
from .transformer_enc_dec import TransformerEncoderDecoder
from .tden import TDENBiTransformer, TDENPretrain

__all__ = list(globals().keys())