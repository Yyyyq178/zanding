export HF_ENDPOINT=https://hf-mirror.com

python eval_metrics.py \
  --hr_dir CelebA-Test-3000/HQ \
  --sr_dir Evaluate_Resume_hard_swinir_old/ariter64-diffstepsddim100-temp0.95-image_num70000_ema_evaluate/sr_images \
  --device cuda:0 \
  --batch_size 8 \
  --out_json json/Evaluate_Resume_hard_swinir_old.json