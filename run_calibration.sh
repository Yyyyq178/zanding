#!/bin/bash

# 显存设置 (如果需要)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Calibration..."

# 运行校准脚本
python calibrate_stats.py \
    --model mar_base \
    --img_size 512 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
<<<<<<< HEAD
    --img_size 512 \
    --resume output_Resume/checkpoint-last.pth \
    --hr_data_path /root/autodl-tmp/zanding/CelebA-Test-400/HQ \
    --lr_data_path /root/autodl-tmp/zanding/CelebA-Test-400/LQ \
    --output_dir pretrained_models/70:40 \
    --calib_batches 50 \
=======
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --num_sampling_steps "ddim100" \
    --resume "output_Resume/checkpoint-last.pth" \
    --hr_data_path "/root/autodl-tmp/zanding/CelebA-Test-400/HQ" \
    --lr_data_path "/root/autodl-tmp/zanding/CelebA-Test-400/LQ" \
    --output_dir "pretrained_models/40_10_traj" \
>>>>>>> X_0
    --batch_size 8 \
    --calib_batches 50 \
    --conf_window "40:10" \
    --use_lr_inject \
    --use_rope \
<<<<<<< HEAD
    --use_mse_loss \
    --conf_window "70:40" \
    --num_sampling_steps "ddim100"
=======
    --use_mse_loss
>>>>>>> X_0
