#!/bin/bash

train_batch_size=${1:-8192}
learning_rate=${2:-"6e-3"}
precision=${3:-"fp16"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-7038}
save_checkpoint_steps=${7:-200}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-128}
num_trainers=${12:-4}
node_rank=${13:-0}
gradient_accumulation_steps_phase2=512
job_name="bert_lamb_pretraining"
CODEDIR=`pwd`
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints
MASTER_NODE= # input your master node's ip
MASTER_PORT= # input your master node's port

mkdir -p $CHECKPOINTS_DIR

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi
ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"

CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir="  #add your own 'Bert wiki-only' train dataset's path
CMD+=" --eval_dir="   #add your own 'Bert wiki-only' eval dataset's path
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --skip_checkpoint=True"
CMD+=" --config_file=bert_config.json"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=12345"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "

CMD="python -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes=$num_trainers --node_rank=$node_rank --master_addr=$MASTER_NODE --master_port=$MASTER_PORT $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished pretraining"