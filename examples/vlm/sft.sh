#!/bin/bash
GPUS_PER_NODE=8
NNODES=$WORLD_SIZE
NODE_RANK=$RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mode="train"
stage="sft"
task="gpt-clip-vlm-sft"
lr=3e-5
warmup_steps=1000
lr_scheduler="cosine"
image_dir="data/llava_next_convert"
image_dir_2="data/llava_recap_convert/"
data_dir="data/llava_merge_processed"
model_path="model/vlm/"
save_path="model/vlm/sft"
max_len=1024
batch_size=720
use_cuda=True
print_every_steps=1
save_every_steps=5000
logger_smooth_loss="ma"
total_steps=25000
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
    --use_checkpoint \
    --raw_image_path $image_dir $image_dir_2 \
    --n_dataloader_workers 8
