#!/bin/bash

# 设置端口 (防止冲突)
export MASTER_PORT=29500

# 运行训练
# 注意参数的对应关系：
# --img_size 512      -> 对应 HR 分辨率
# --buffer_size 64    -> 对应 LR 分辨率 128 (128/16)^2 = 64
# --hr_data_path      -> 指向你的 HR 数据集根目录

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_mar.py \
    --model mar_base \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --img_size 512 \
    --buffer_size 64 \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --cfg 1.0 \
    --epochs 100 \
    --warmup_epochs 5 \
    --batch_size 8 \
    --grad_clip 3.0 \
    --blr 3.0e-4 \
    --hr_data_path /root/autodl-tmp/zanding/dataset/HR_image \
    --output_dir output_sr_train \
    --eval_freq 2 \
    --save_last_freq 2 \
    --eval_bsz 8 \
    --online_eval