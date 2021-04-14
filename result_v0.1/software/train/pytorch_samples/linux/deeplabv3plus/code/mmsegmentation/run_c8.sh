export PYTHONPATH=$PYTHONPATH:`pwd`/mmseg
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export TOP_VAL=78.5
python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
               tools/train.py configs/fp16/deeplabv3plus_r50-d8_512x1024_80k_fp16_cityscapes_c8.py \
				--launcher pytorch --checksum 37724b19b6e5d41f9f147936d60b3c29 
