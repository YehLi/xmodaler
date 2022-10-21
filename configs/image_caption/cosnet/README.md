# Introduction
This repository is for **Comprehending and Ordering Semantics for Image Captioning** (CVPR 2022).

Please cite with the following BibTeX:

```
@inproceedings{cosnet2022cvpr,
  title={Comprehending and Ordering Semantics for Image Captioning},
  author={Li, Yehao and Pan, Yingwei and Yao, Ting and Mei, Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Data preparation
1. Download the cosnet [Baidu](https://pan.baidu.com/s/1x3BJzemXcIvKo1padRq_cg)/[GoogleDriver](https://drive.google.com/drive/folders/1zgC2ddIt5aMxo07DoIiJgMTqNhGlZKXp?usp=sharing) folder and put it into open_source_dataset/mscoco_dataset

2. Download the CLIP_RN101_49 [Baidu](https://pan.baidu.com/s/1S-YVjumU7fK6atzhrE_1yg)/[GoogleDriver](https://drive.google.com/drive/folders/12WbB3VkJh0NB9b8xlflY3PxTM3BwHtnF?usp=sharing) folder and put it into open_source_dataset/mscoco_dataset/features

3. Models and results can be downloaded here [Baidu](https://pan.baidu.com/s/1FESU3-pgTRYvsLo9hBfzqg)/[GoogleDriver](https://drive.google.com/drive/folders/1CjuCLm-MEhOLY7E7sD2kbTm2jxCxIYVc?usp=sharing)

Access code for Baidu is **cosn**

## Training
### Train COS-Net model
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file ./configs/image_caption/cosnet/cosnet.yaml
```

### Train COS-Net model using self critical
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file ./configs/image_caption/cosnet/cosnet_rl.yaml
```
