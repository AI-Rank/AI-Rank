export PYTHONPATH=$PYTHONPATH:`pwd`/mmseg
export CUDA_VISIBLE_DEVICES=1
export TOP_VAL=78.5
python3 tools/train.py configs/fp16/deeplabv3plus_r50-d8_512x1024_80k_fp16_cityscapes.py --checksum 37724b19b6e5d41f9f147936d60b3c29 

