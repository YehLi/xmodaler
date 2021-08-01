"""	
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.
from .build import (
    build_xmodaler_train_loader,
    build_xmodaler_valtest_loader
)

from .common import DatasetFromList, MapDataset
from .images.mscoco import MSCoCoDataset
from .images.mscoco_bert import MSCoCoBertDataset
from .images.conceptual_captions import ConceptualCaptionsDataset
from .images.vqa import VQADataset
from .images.flickr30k import Flickr30kDataset
from .videos.msvd import MSVDDataset
from .videos.msrvtt import MSRVTTDataset


__all__ = [k for k in globals().keys() if not k.startswith("_")]