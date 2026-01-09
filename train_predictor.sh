#!/bin/bash

# 训练置信度预测器 (Confidence Predictor)
# 使用成对的 HR 和 LR 数据集 (CelebA-Test-400)

python train_predictor.py \
    --hr_path /root/autodl-tmp/zanding/CelebA-Test-400/HQ \
    --lr_path /root/autodl-tmp/zanding/CelebA-Test-400/LQ \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --output_dir output_predictor \
    --img_size 512 \
    --epochs 150 \
    --batch_size 8