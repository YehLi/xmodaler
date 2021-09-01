# Use Builtin Datasets

A dataset can be used by wrapping it into a torch Dataset. This document explains how to setup the builtin datasets so they can be used by X-modaler. The annotations for builtin datasets can be downloaded [here](https://drive.google.com/drive/folders/1vx9n7tAIt8su0y_3tsPJGvMPBMm8JLCZ).

X-modaler has builtin supports for a few datasets (e.g., MSCOCO or MSVD). The corresponding dataset wrappers are provided in `xmodaler/datasets`:
```
xmodaler/datasets/
  images/
    mscoco.py
  videos/
    msvd.py  
```
You can specify which dataset wrapper to use by `DATASETS.TRAIN`, `DATASETS.VAL` and `DATASETS.TEST` in the config file. 

## Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
mscoco_dataset/
  mscoco_caption_anno_train.pkl
  mscoco_caption_anno_val.pkl
  mscoco_caption_anno_test.pkl
  vocabulary.txt
  captions_val5k.json
  captions_test5k.json
  # image files that are mentioned in the corresponding json
  features/
    up_down/
      *.npz
```

## Expected dataset structure for [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/):

```
msvd_dataset/
  msvd_caption_anno_train.pkl
  msvd_caption_anno_val.pkl
  msvd_caption_anno_test.pkl
  vocabulary.txt
  captions_val.json
  captions_test.json
  # videos files that are mentioned in the corresponding json
  features/
    resnet152/
      *.npy
```

## Expected dataset structure for [MSR-VTT](http://ms-multimedia-challenge.com/2017/dataset):

```
msrvtt_dataset/
  msrvtt_caption_anno_train.pkl
  msrvtt_caption_anno_val.pkl
  msrvtt_caption_anno_test.pkl
  vocabulary.txt
  captions_val.json
  captions_test.json
  # videos files that are mentioned in the corresponding json
  msrvtt_torch/
    feature/
	  resnet152/
	    *.npy
```

When the dataset wrapper and data files are ready, you need to specify the corresponding paths to these data files in the config file. For example, 
```
DATALOADER:
	FEATS_FOLDER: '../open_source_dataset/mscoco_dataset/features/up_down'    # feature folder
	ANNO_FOLDER: '../open_source_dataset/mscoco_dataset' # annotation folders
INFERENCE:
	VOCAB: '../open_source_dataset/mscoco_dataset/vocabulary.txt'
	VAL_ANNFILE: '../open_source_dataset/mscoco_dataset/captions_val5k.json'
	TEST_ANNFILE:  '../open_source_dataset/mscoco_dataset/captions_test5k.json'
```
