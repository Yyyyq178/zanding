export MASTER_PORT=29505

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_mar.py \
    --model mar_base \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --vae_path pretrained_models/vae/kl-f16.ckpt \
    --buffer_size 64 \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --batch_size 1 \
    --eval_bsz 16 \
    --img_size 512 \
    --resume output_sr_train_diffusionloss_codeformer_RoPE_3_1:5_swinir_kl-f16 \
    --evaluate \
    --paired_test \
    --hr_data_path /root/autodl-tmp/zanding/CelebA-Test-3000/HQ \
    --lr_data_path /root/autodl-tmp/zanding/CelebA-Test-3000/LQ \
    --use_swinir \
    --swinir_ckpt pretrained_models/swinir/face_swinir_v1.ckpt \
    --swinir_batch 4 \
    --output_dir output_test_paired