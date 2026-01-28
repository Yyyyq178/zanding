#!/bin/bash
export MASTER_PORT=29501

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_mar.py \
    --model mar_base \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --epochs 600 \
    --batch_size 8 \
    --grad_clip 1.0 \
    --cfg_drop_prob 0.1 \
    --steps_per_epoch 250 \
    --lr 5.0e-4 \
    --hr_data_path /root/autodl-tmp/zanding/data \
    --val_data_path /root/autodl-tmp/zanding/data \
    --output_dir output_Resume_harder \
    --degradation codeformer \
    --eval_freq 4 \
    --save_last_freq 2 \
    --eval_bsz 8 \
    --img_size 512 \
    --sche lin0 \
    --multi_scale \
    --use_lr_inject \
    --use_rope \
    --use_mse_loss \
    --online_eval