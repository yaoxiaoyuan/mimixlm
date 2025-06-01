#!/bin/bash
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=23045

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mode="train"
stage="pretrain"
task_name="tinystories-small-pretrain"
lr=1e-3
warmup_steps=500
lr_scheduler="cosine"
data_dir="data/tinystories_processed/"
model_path="model/tinystories_small/init"
save_path="model/tinystories_small/"
max_len=512
batch_size=180
use_cuda=True
print_every_steps=1
save_every_steps=5000
logger_smooth_loss="ma"
total_steps=15000

torchrun $DISTRIBUTED_ARGS mimixlm.py \
    --mode $mode \
    --stage $stage \
    --task_name $task_name \
    --train_data_path $data_dir \
    --model_path $model_path \
    --lr $lr \
    --warmup_steps $warmup_steps \
    --lr_scheduler $lr_scheduler \
    --max_len $max_len \
    --batch_size $batch_size \
    --print_every_steps $print_every_steps \
    --use_cuda \
    --save_path $save_path \
    --save_every_steps $save_every_steps \
    --logger_smooth_loss $logger_smooth_loss \
    --use_deepspeed \
    --accumulate_steps 1 \
    --total_steps $total_steps \
    --use_checkpoint
