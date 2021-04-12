#!/bin/bash

if [ ! -d "./output/checkpoint/sync" ]; then
    mkdir -p ./output/checkpoint/sync
fi

if [ ! -d "./output/checkpoint/async" ]; then
    mkdir -p ./output/checkpoint/async
fi

if [ ! -d "./log/sync" ]; then
    mkdir -p ./log/sync
fi

if [ ! -d "./log/async" ]; then
    mkdir -p ./log/async
fi

export PADDLE_PSERVERS_IP_PORT_LIST=127.0.0.1:36001,127.0.0.1:36002
export PADDLE_WORKERS_IP_PORT_LIST=127.0.0.1:36006,127.0.0.1:36007
trainer_nums=2
pserver_nums=2


for((i=0;i<${pserver_nums};i++)) 
do
    export TRAINING_ROLE=PSERVER
    export PADDLE_TRAINER_ID=$i
    python -u deepfm_distribute.py --sync_mode=False  &> ./log/${mode}/pserver.$i.log &
done

for((i=0;i<${trainer_nums};i++))
do
    export TRAINING_ROLE=TRAINER
    export PADDLE_TRAINER_ID=$i
    python -u deepfm_distribute.py --sync_mode=False &> ./log/${mode}/worker.$i.log &
done
