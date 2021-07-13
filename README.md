# X-modaler
X-modaler is a versatile and high-performance codebase for cross-modal analytics. This codebase unifies comprehensive high-quality modules in state-of-the-art vision-language techniques, which are organized in a standardized and user-friendly fashion. The annotations can be downloaded [here](https://drive.google.com/drive/folders/1oDjSwScWztViv--aJzoDNKkh2txqBArV?usp=sharing).

# License
FaceX-Zoo is released under the [Apache License, Version 2.0](LICENSE).

## Getting Started with X-modaler

This document provides a brief intro of the usage of builtin command-line tools in X-modaler, showing how to run inference with an existing model, and how to train a builtin model on a custom dataset (e.g., MSCOCO or MSVD).

### Training & Evaluation in Command Line

We provide a script in "train_net.py", that is made to train all the configs provided in X-modaler. You may want to use it as a reference to write your own training script.

To train a model with "train_net.py", first setup the corresponding datasets following [xmodaler/datasets/README.md](./xmodaler/datasets/README.md), then run:
```
python train_net.py --num-gpus 4 \
 	--config-file configs/COCO/updown.yaml
```