#!/bin/bash
 
#GPUS_PER_NODE=8
#NNODES=$WORLD_SIZE
#NODE_RANK=$RANK
#MASTER_ADDR=$MASTER_ADDR

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR="localhost"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mode="train"
stage="dpo"
task="large-dpo-2048"
lr=1e-5
warmup_steps=500
lr_scheduler="cosine"
data_dir="data/ultrafeedback_processed/"
model_path="model/gpt_large_sft"
save_path="model/gpt_large_dpo"
max_len=2048
batch_size=32
use_cuda=True
print_every_steps=1
save_every_steps=2000
logger_smooth_loss="ma"
total_steps=5700

torchrun $DISTRIBUTED_ARGS mimixlm.py \
    --mode $mode \
    --stage $stage \
    --task $task \
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
    --dpo_beta 0.1 \
    --use_sft_loss \
    --dpo_lambda 0.1 \
    --use_checkpoint
