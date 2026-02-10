export HF_ENDPOINT=https://hf-mirror.com

python eval_metrics.py \
  --sr_dir Evaluate_Resume_WebPhoto/ariter64-diffstepsddim100-temp0.95-image_num70000_ema_evaluate/sr_images \
  --device cuda:0 \
  --batch_size 8 \
  --fid_stats pretrained_models/ffhq_stats.npz \
  --out_json json/Evaluate_Resume_WebPhoto.json
  #--hr_dir CelebA-Test-3000-new/HR\