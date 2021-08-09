# XE
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file ./configs/image_caption/updown/updown.yaml

# RL
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file ./configs/image_caption/updown/updown_rl.yaml