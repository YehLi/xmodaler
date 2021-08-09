# XE
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file ./configs/image_caption/transformer/transformer.yaml

# RL
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file ./configs/image_caption/transformer/transformer_rl.yaml