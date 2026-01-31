export MASTER_PORT=29504
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_mar.py \
    --model mar_base \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --batch_size 1 \
    --eval_bsz 16 \
    --img_size 512 \
    --resume output_Resume_harder_swinir \
    --evaluate \
    --temperature 0.95 \
    --paired_test \
    --hr_data_path CelebA-Test-3000-new/HR \
    --lr_data_path CelebA-Test-3000-new/LR \
    --use_swinir \
    --swinir_ckpt pretrained_models/swinir/swinir_restoration512_L1.pth \
    --use_lr_inject \
    --use_rope \
    --use_mse_loss \
    --use_dynamic_maskgit \
    --conf_threshold 1.0 \
    --conf_pmin 0.05 \
    --conf_method stats \
    --conf_window '40:10' \
    --output_dir Evaluate_Resume_hard_swinir
    #--cfg_scale 1.2 \
    #--predictor_ckpt output_predictor/predictor_latest.pth \
    