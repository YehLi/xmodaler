# random negative sample
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file ./configs/mm_understanding/flickr30k_retrieval/uniter/flickr30k_retrieval.yaml

# hard negative mining
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 --config-file ./configs/mm_understanding/flickr30k_retrieval/uniter/flickr30k_retrieval_hard.yaml
