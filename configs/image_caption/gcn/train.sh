# XE
python3 train_net.py --num-gpus 4 --config-file ./configs/image_caption/gcn/gcn.yaml

# RL
CUDA_VISIBLE_DEVICES=3,2,1,0 GLOO_SOCKET_IFNAME=eth0 NCCL_NSOCKS_PERTHREAD=4 NCCL_SOCKET_NTHREADS=4 python3 train_net.py --num-gpus 4 \
    --config-file ./configs/image_caption/gcn/gcn_rl.yaml \
    --num-machines 3 \
    --machine-rank 0 \
    --dist-url "tcp://10.207.174.48:12346" \
