# X-modaler
[X-modaler](https://xmodaler.readthedocs.io/en/latest/) is a versatile and high-performance codebase for cross-modal analytics. This codebase unifies comprehensive high-quality modules in state-of-the-art vision-language techniques, which are organized in a standardized and user-friendly fashion.

## Installation
See [installation instructions](https://xmodaler.readthedocs.io/en/latest/tutorials/installation.html).

### Requiremenets
* Linux or macOS with Python ≥ 3.6
* PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this
* fvcore
* pytorch_transformers
* jsonlines
* pycocotools

## Getting Started 
See [Getting Started with X-modaler](https://xmodaler.readthedocs.io/en/latest/tutorials/getting_started.html)

### Training & Evaluation in Command Line

We provide a script in "train_net.py", that is made to train all the configs provided in X-modaler. You may want to use it as a reference to write your own training script.

To train a model(e.g., UpDown) with "train_net.py", first setup the corresponding datasets following [datasets](xmodaler/datasets/README.md), then run:
```
# Teacher Force
python train_net.py --num-gpus 4 \
 	--config-file configs/image_caption/updown.yaml

# Reinforcement Learning
python train_net.py --num-gpus 4 \
 	--config-file configs/image_caption/updown_rl.yaml
```

# Model Zoo and Baselines
A large set of baseline results and trained models are available [here](https://xmodaler.readthedocs.io/en/latest/notes/benchmarks.html).

## API Documentation
* xmodaler.checkpoint
* xmodaler.config
* xmodaler.datasets
* xmodaler.engine
* xmodaler.evaluation
* xmodaler.functional
* xmodaler.losses
* xmodaler.lr_scheduler
* xmodaler.modeling
* xmodaler.optim
* xmodaler.scorer
* xmodaler.tokenization
* xmodaler.utils

More about our [API Documentation](https://xmodaler.readthedocs.io/en/latest/modules/index.html)

## License
X-modaler is released under the [Apache License, Version 2.0](LICENSE).

## Citing X-modaler
If you use X-modaler in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{Xmodaler2021,
  author =       {Yehao Li, Yingwei Pan, Jingwen Chen, Ting Yao, and Tao Mei},
  title =        {X-modaler: A Versatile and High-performance Codebase for Cross-modal Analytics},
  booktitle =    {Proceedings of the 29th ACM international conference on Multimedia},
  year =         {2021}
}
```
