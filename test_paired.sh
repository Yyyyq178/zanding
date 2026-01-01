export MASTER_PORT=29505

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_mar.py \
    --model mar_base \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --buffer_size 64 \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --batch_size 1 \
    --eval_bsz 8 \
    --img_size 512 \
    --resume output_Resume \
    --evaluate \
    --paired_test \
    --hr_data_path /root/autodl-tmp/zanding/CelebA-Test-300/HQ \
    --lr_data_path /root/autodl-tmp/zanding/CelebA-Test-300/LQ \
    --use_lr_inject \
    --use_rope \
    --use_mse_loss \
    --output_dir Evaluate_Resume_Baseline_8

