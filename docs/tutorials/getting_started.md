## Getting Started with X-modaler

This document provides a brief intro of the usage of builtin command-line tools in X-modaler, showing how to run inference with an existing model, and how to train a builtin model on a custom dataset (e.g., MSCOCO or MSVD).

### Training & Evaluation in Command Line

We provide a script in "train_net.py", which is made to train all the configs provided in X-modaler. You may want to use it as a reference to write your own training script.

To train a model(e.g., UpDown) with "train_net.py", first setup the corresponding datasets following [datasets](using_builtin_datasets.md), then run:
```
# Teacher Force
python train_net.py --num-gpus 4 \
 	--config-file configs/image_caption/updown.yaml

# Reinforcement Learning
python train_net.py --num-gpus 4 \
 	--config-file configs/image_caption/updown_rl.yaml
```
