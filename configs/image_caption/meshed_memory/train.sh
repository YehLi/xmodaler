# XE
CUDA_VISIBLE_DEVICES=0 python3 train_net.py --num-gpus 1 --config-file configs/image_caption/meshed_memory/meshed_memory.yaml

# RL
CUDA_VISIBLE_DEVICES=0 python3 train_net.py --num-gpus 1 --config-file configs/image_caption/meshed_memory/meshed_memory_rl.yaml