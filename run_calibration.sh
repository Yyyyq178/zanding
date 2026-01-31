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
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --num_sampling_steps "ddim100" \
    --degradation codeformer \
    --resume "output_Resume_harder_swinir/checkpoint-last.pth" \
    --hr_data_path "CelebA-Test-3000-new/HR" \
    --lr_data_path "CelebA-Test-3000-new/LR" \
    --output_dir "pretrained_models/40_10_harder_swinir" \
    --batch_size 16 \
    --calib_batches 500 \
    --conf_window "40:10" \
    --temperature 0.95 \
    --use_swinir \
    --swinir_ckpt pretrained_models/swinir/swinir_restoration512_L1.pth \
    --use_lr_inject \
    --use_rope \
    --use_mse_loss
    # --cfg_scale 1.2 \
