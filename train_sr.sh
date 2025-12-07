#!/bin/bash
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
# 设置端口 (防止冲突)
export MASTER_PORT=29501

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
    --buffer_size 64 \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --cfg 1.0 \
    --epochs 800 \
    --warmup_epochs 10 \
    --batch_size 2 \
    --grad_clip 1.0 \
    --steps_per_epoch 250 \
    --blr 4.0e-2 \
    --hr_data_path /root/autodl-tmp/zanding/data \
    --val_data_path /root/autodl-tmp/zanding/data \
    --output_dir output_sr_train_diffusionloss_codeformer_RoPE_1 \
    --degradation codeformer \
    --eval_freq 2 \
    --save_last_freq 2 \
    --eval_bsz 2 \
    --img_size 512 \
    --multi_scale \
    --lr_schedule cosine \
    --online_eval
    #--grad_checkpointing \
    #--resume output_sr_train_diffusionloss_codeformer_RoPE \
    