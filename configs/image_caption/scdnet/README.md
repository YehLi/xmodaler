# Semantic-Conditional Diffusion Networks for Image Captioning

**For compatibility reasons, the reimplemented code and more detailed information are released in an official independent [repository](https://github.com/jianjieluo/SCD-Net) currently. Code of SCD-Net will be merged into this official X-modaler repository in the future.**

## Introduction
This is the official repository for [**Semantic-Conditional Diffusion Networks for Image Captioning**](https://arxiv.org/abs/2212.03099)(SCD-Net). SCD-Net is a cascaded diffusion captioning model with a novel semantic-conditional diffusion process that upgrades conventional diffusion model with additional semantic prior. A novel guided self-critical sequence training strategy is further devised to stabilize and boost the diffusion process. 

To our best knowledge, SCD-Net is the first diffusion-based captioning model that achieves better performance than the naive auto-regressive transformer captioning model **conditioned on the same visual features(i.e. [bottom-up attention region features](https://github.com/peteanderson80/bottom-up-attention)) in both XE and RL training stages.** SCD-Net is also **the first diffusion-based captioning model that adopts CIDEr-D optimization successfully** via a novel guided self-critical sequence training strategy. 

SCD-Net achieves state-of-the-art performance among non-autoregressive/diffusion captioning models and comparable performance aginst the state-of-the-art autoregressive captioning models, which indicates the promising potential of using diffusion models in the challenging image captioning task.


## Framework
![scdnet](imgs/scdnet.png)


## Citation
If you use this code for your research, please cite:

```
@article{luo2022semantic,
  title={Semantic-Conditional Diffusion Networks for Image Captioning},
  author={Luo, Jianjie and Li, Yehao and Pan, Yingwei and Yao, Ting and Feng, Jianlin and Chao, Hongyang and Mei, Tao},
  journal={arXiv preprint arXiv:2212.03099},
  year={2022}
}
```