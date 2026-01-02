#!/bin/bash

# 显存设置 (如果需要)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Calibration..."

# 运行校准脚本
python calibrate_stats.py \
    --model mar_base \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --buffer_size 64 \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --img_size 512 \
    --resume output_Resume/checkpoint-last.pth \
    --hr_data_path /root/autodl-tmp/zanding/CelebA-Test-400/HQ \
    --lr_data_path /root/autodl-tmp/zanding/CelebA-Test-400/LQ \
    --output_dir pretrained_models/70:40 \
    --calib_batches 50 \
    --batch_size 8 \
    --use_lr_inject \
    --use_rope \
    --use_mse_loss \
    --conf_window "70:40" \
    --num_sampling_steps "ddim100"
