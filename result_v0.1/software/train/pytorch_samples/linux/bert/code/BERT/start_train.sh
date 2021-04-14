#!/bin/bash
set -xe
export LD_LIBRARY_PATH=/usr/lib/libibverbs/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

batch_size=96     # batch size per gpu
num_gpus=$1       # number of gpu per node
precision=$2      # fp32 | fp16
gradient_accumulation_steps=$(expr 67584 \/ $batch_size \/ $num_gpus)
train_batch_size=$(expr 67584 \/ $num_gpus)   # total batch_size per gpu
train_steps=$3    # max train steps
num_trainers=$4   # number of nodes
node_rank=$5      # current node rank

bash run_benchmark_multi_nodes.sh $train_batch_size 6e-3 $precision $num_gpus 0.2843 $train_steps 200 false true true $gradient_accumulation_steps $num_trainers $node_rank
