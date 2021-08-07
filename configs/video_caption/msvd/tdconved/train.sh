CUDA_VISIBLE_DEVICES=0 python3 train_net.py --config-file configs/video_caption/msvd/tdconved/tdconved.yaml --num-gpus 1 OUTPUT_DIR ./experiments/msvd-tdconved DATALOADER.MAX_FEAT_NUM 50
