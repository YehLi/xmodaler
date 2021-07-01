#export PATH=$PATH:/media/winne/DATA2/new_code/opensource/coco_caption
#CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING="1" python train_net.py --num-gpus 1 --config-file ./configs/salstm.yaml
python test_net.py --num-gpus 1 --config-file ./configs/salstm.yaml
