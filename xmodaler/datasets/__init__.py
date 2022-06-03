"""	
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.
from .build import (
    build_xmodaler_train_loader,
    build_xmodaler_valtest_loader,
    build_dataset_mapper
)

from .common import DatasetFromList, MapDataset
from .images.mscoco import MSCoCoDataset, MSCoCoSampleByTxtDataset
from .images.mscoco_bert import MSCoCoBertDataset
from .images.mscoco_cosnet import MSCoCoCOSNetDataset
from .images.mscoco_feat import MSCoCoFeatDataset
from .images.mscoco_raw import MSCoCoRawDataset
from .images.conceptual_captions import ConceptualCaptionsDataset, ConceptualCaptionsDatasetForSingleStream
from .images.vqa import VQADataset
from .images.vcr import VCRDataset
from .images.flickr30k import Flickr30kDataset
from .images.flickr30k_single_stream import Flickr30kDatasetForSingleStream, Flickr30kDatasetForSingleStreamVal
from .videos.msvd import MSVDDataset
from .videos.msrvtt import MSRVTTDataset


__all__ = [k for k in globals().keys() if not k.startswith("_")]
