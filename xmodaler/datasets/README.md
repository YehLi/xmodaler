# Use Builtin Datasets

A dataset can be used by wrapping it into a torch Dataset. This document explains how to setup the builtin datasets so they can be used by XModaler.

XModaler has builtin support for a few datasets (e.g., MSCOCO or MSVD). The corresponding dataset wrappers are provided in `./xmodaler/datasets`:
```
xmodaler/datasets/
  image_caption/
    mscoco.py
  video_caption/
    msvd.py  
```
You can specify which dataset wrapper to use by `DATASETS.TRAIN` and `DATASETS.TEST` in the config file. The [model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) contains configs and models that use these builtin datasets.

## Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
mscoco_open_source/
  mscoco_caption_anno_train.pkl
  mscoco_caption_anno_val.pkl
  mscoco_caption_anno_test.pkl
  coco_vocabulary.txt
  captions_val5k.json
  captions_test5k.json
  # image files that are mentioned in the corresponding json
mscoco_torch/
  feature/
    up_down_100/
      *.npz
```
We provide a script `tools/coco_preprocess.py` to generate the above files based on the downloaded original dataset. Then, MSCOCO dataset is wrapped by `class MSCoCoDatasetMapper` in the script `xmodaler/datasets/image_caption/mscoco.py` for both training and evaluation. You may want to use it as a reference to write your own dataset wrapper for image captioning.

## Expected dataset structure for [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/):

```
msvd_open_source/
  msvd_caption_anno_train.pkl
  msvd_caption_anno_val.pkl
  msvd_caption_anno_test.pkl
  msvd_vocabulary.txt
  captions_val_cocostyle.json
  captions_test_cocostyle.json
  # videos files that are mentioned in the corresponding json
msvd_torch/
  feature/
    resnet152/
      *.npy
```
We provide a script `tools/msvd_preprocess.py` to generate the above files based on the downloaded original dataset. Then, MSVD dataset is wrapped by `class MSVDDatasetMapper` in the script `xmodaler/datasets/video_caption/msvd.py` for both training and evaluation. You may want to use it as a reference to write your own dataset wrapper for video captioning.

## Expected dataset structure for [MSR-VTT](http://ms-multimedia-challenge.com/2017/dataset):

```
msrvtt_open_source/
  msrvtt_caption_anno_train.pkl
  msrvtt_caption_anno_val.pkl
  msrvtt_caption_anno_test.pkl
  msrvtt_vocabulary.txt
  captions_val_cocostyle.json
  captions_test_cocostyle.json
  # videos files that are mentioned in the corresponding json
msrvtt_torch/
  feature/
    resnet152/
      *.npy
```
We provide a script `tools/msrvtt_preprocess.py` to generate the above files based on the downloaded original dataset. Then, MSVD dataset is wrapped by `class MSRVTTDatasetMapper` in the script `xmodaler/datasets/video_caption/msrvtt.py` for both training and evaluation.

When the dataset wrapper and data files are ready, you need to specify the corresponding paths to these data files in the config file. For example, 
```
DATALOADER:
	FEATS_FOLDER: 'mscoco_torch/feature/up_down_100'    # feature folder
	ANNO_FILE: 'mscoco_open_source/mscoco_caption_anno' # file prefix
INFERENCE:
	VOCAB: 'mscoco_open_source/coco_vocabulary.txt'
	VAL_ANNFILE: 'mscoco_open_source/captions_val5k.json'
	TEST_ANNFILE:  'mscoco_open_source/captions_test5k.json'
```

