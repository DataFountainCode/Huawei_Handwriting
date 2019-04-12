CUDA_VISIBLE_DEVICES=${1} python -m pdb tools/train.py configs/faster_rcnn_r50_fpn_hanzishibie.py  --validate --gpus 1 
