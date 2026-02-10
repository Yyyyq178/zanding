import math
import sys
from typing import Iterable

import torch
import torch.distributed as dist

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
from dataset.codeformer_face import CodeFormerDegradation as CodeFormerDegradationFace
from dataset.realesrgan_natural import RealESRGANDegradationNatural
import pyiqa
import shutil
import cv2
import numpy as np
import os
import copy
import time

class _MetricRegistry:
    """Cache and own expensive pyiqa metrics on a single (rank 0) process."""

    def __init__(self, device: str):
        self.device = device
        self.metric_specs = {
            # full-reference metrics (need hr_tensor)
            'psnr': ('psnr', True),
            'ssim': ('ssim', True),
            'lpips': ('lpips', True),
            'dists': ('dists', True),
            # 'ms_ssim': ('ms_ssim', True),
            # no-reference metrics
            'musiq': ('musiq', False),
            # 'brisque': ('brisque', False),
            # 'maniqa': ('maniqa', False),
            # 'clipiqa': ('clipiqa', False),
            # 'niqe': ('niqe', False),
            # 'piqe': ('piqe', False),
        }
        print("Initializing evaluation metrics on rank0 (cached across runs)...")
        self.fid_metric = pyiqa.create_metric('fid', device=self.device)
        self.full_ref_metrics = {}
        self.no_ref_metrics = {}

        for log_name, (metric_name, is_full_ref) in self.metric_specs.items():
            metric = pyiqa.create_metric(metric_name, device=self.device)
            if is_full_ref:
                self.full_ref_metrics[log_name] = metric
            else:
                self.no_ref_metrics[log_name] = metric


_METRIC_REGISTRY = None


def _get_or_create_metrics(device: str):
    """Create metrics on rank0 only and reuse them between evaluate calls."""

    global _METRIC_REGISTRY
    if not misc.is_main_process():
        return None
    if _METRIC_REGISTRY is None or _METRIC_REGISTRY.device != device:
        _METRIC_REGISTRY = _MetricRegistry(device)
    else:
        print("Reusing cached evaluation metrics on rank0.")
    return _METRIC_REGISTRY


def _gather_tensor(tensor: torch.Tensor):
    if not misc.is_dist_avail_and_initialized():
        return tensor
    tensor_list = [torch.zeros_like(tensor) for _ in range(misc.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def preprocess_with_swinir(lr_tensor, swinir_model, mini_batch_size=4):
    """Clean LR images with SwinIR and map values back to [-1, 1]."""
    if swinir_model is None:
        return lr_tensor

    lr_input = (lr_tensor + 1) / 2  # [-1,1] -> [0,1]
    cleaned_batches = []

    for start in range(0, lr_input.size(0), mini_batch_size):
        mini_input = lr_input[start:start + mini_batch_size]
        cleaned_batches.append(swinir_model(mini_input))

    lr_cleaned = torch.cat(cleaned_batches, dim=0)
    lr_tensor = (lr_cleaned * 2) - 1  # [0,1] -> [-1,1]

    return lr_tensor.clamp(-1, 1)


def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    swinir_model=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    num_steps_per_epoch = len(data_loader)
    if args.steps_per_epoch > 0 and args.steps_per_epoch < len(data_loader):
        num_steps_per_epoch = args.steps_per_epoch

    if args.degradation == 'realesrgan_natural':
        degradation_model = RealESRGANDegradationNatural()
        if misc.get_rank() == 0 and epoch == 0:
            print("Using Natural Degradation: Real-ESRGAN (High-Order)")
    else:
        degradation_model = CodeFormerDegradationFace()

    gate_multiplier = 1.0

    for data_iter_step, (samples_hr, samples_lr, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if args.steps_per_epoch > 0 and data_iter_step >= args.steps_per_epoch:
            break

        lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_steps_per_epoch + epoch, args)

        samples_hr = samples_hr.to(device, non_blocking=True)

        samples_lr = degradation_model(samples_hr, scale=None)
        samples_lr = samples_lr.to(device, non_blocking=True)

        if swinir_model is not None:
            samples_lr = preprocess_with_swinir(samples_lr, swinir_model, args.swinir_batch)

        with torch.no_grad():
            if args.use_cached:
                raise NotImplementedError("Cached mode not supported for SR yet.")
            posterior_hr = vae.encode(samples_hr)
            x_hr = posterior_hr.sample().mul_(0.2325)

            posterior_lr = vae.encode(samples_lr)
            x_lr = posterior_lr.sample().mul_(0.2325)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model_out = model(
                x_hr, x_lr, gate_multiplier=gate_multiplier,
            )
            loss, loss_diff, loss_mse = model_out

            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_diff=loss_diff.item())
        metric_logger.update(loss_mse=loss_mse.item())

        if args.use_lr_inject:
            model_ref = model.module if hasattr(model, "module") else model
            gate_mean = model_ref.get_lr_inject_gate_mean()
            metric_logger.update(lr_inject_gate_mean=gate_mean)
            metric_logger.update(lr_inject_gate_multiplier=gate_multiplier)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_diff_mean = misc.all_reduce_mean(loss_diff.item())
        loss_mse_mean = misc.all_reduce_mean(loss_mse.item())
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_diff', loss_diff_mean, epoch_1000x)
            log_writer.add_scalar('train_loss_mse', loss_mse_mean, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None,
             use_ema=True, data_loader=None, swinir_model=None, paired_mode=False):
    torch.cuda.empty_cache()
    prev_training_mode = model_without_ddp.training
    model_without_ddp.eval()
    
    enable_metrics = (not args.evaluate) and (not getattr(args, 'only_lr_test', False))

    save_folder = os.path.join(
        args.output_dir,
        "ariter{}-diffsteps{}-temp{}-image_num{}".format(
            args.num_iter,
            args.num_sampling_steps,
            args.temperature,
            args.num_images,
        )
    )
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)

    if misc.get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)

    # 初始化指标 (仅当 enable_metrics 为 True)
    metrics_registry = None
    metric_logger = misc.MetricLogger(delimiter="  ")
    
    if enable_metrics:
        metrics_registry = _get_or_create_metrics(args.device)
        metric_names = ['psnr', 'ssim', 'lpips', 'dists', 'musiq', 'fid']
        for name in metric_names:
            metric_logger.add_meter(name, misc.SmoothedValue())

    # Switch to EMA
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    if data_loader is None:
        print("No data loader provided for evaluation, skipping.")
        return
    
    if not paired_mode:
        if args.degradation == 'realesrgan_natural':
            DegradationClass = RealESRGANDegradationNatural
        else:
            DegradationClass = CodeFormerDegradationFace

    sr_save_dir = os.path.join(save_folder, "sr_images")
    hr_save_dir = os.path.join(save_folder, "hr_images")

    print(f"Start evaluation on {len(data_loader)} batches...")

    total_inference_time = 0.0
    total_images_processed = 0
    total_steps_accum = 0

    for i, (imgs_hr, imgs_lr, filenames) in enumerate(data_loader):

        imgs_hr = imgs_hr.cuda(args.device, non_blocking=True)
        imgs_lr = imgs_lr.cuda(args.device, non_blocking=True)

        # if i >= 5: 
        #     print("Finished 5 batches preview, stopping evaluation.")
        #     break 
        
        if not paired_mode:
            if args.evaluate:
                degradation_model = DegradationClass()
                imgs_lr = degradation_model(imgs_hr, scale=None)
            else:
                degradation_model = DegradationClass()
                imgs_lr = degradation_model(imgs_hr, scale=4.0)
        
        # --- 1. 备份原始 LR (用于四张图拼接的最左侧) ---
        imgs_lr_raw = imgs_lr.clone()

        # --- 2. SwinIR 处理 (处理后的 imgs_lr 将作为模型输入及第二张图) ---
        if swinir_model is not None:
            imgs_lr = preprocess_with_swinir(imgs_lr, swinir_model, args.swinir_batch)

        start_time = time.time()
        
        # Inference
        with torch.no_grad():
            posterior_lr = vae.encode(imgs_lr)
            x_lr = posterior_lr.sample().mul_(0.2325) 

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sampled_tokens, real_steps = model_without_ddp.sample_tokens(
                    bsz=imgs_lr.shape[0], 
                    num_iter=args.num_iter, 
                    x_lr=x_lr,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                )
                sampled_images = vae.decode(sampled_tokens / 0.2325)

        end_time = time.time()
        batch_time = end_time - start_time
        total_inference_time += batch_time
        curr_bsz = imgs_lr.shape[0]
        total_images_processed += curr_bsz
        total_steps_accum += real_steps * curr_bsz

        if misc.get_rank() == 0:
            print(f"[{i}/{len(data_loader)}] "
                  f"文件名: {filenames[0] if len(filenames)>0 else 'N/A'} | "
                  f"推理步数: {real_steps} | "
                  f"Batch耗时: {batch_time:.3f}s | "
                  f"单张平均: {batch_time/curr_bsz:.3f}s")

        sr_tensor = (sampled_images + 1) / 2
        sr_tensor = sr_tensor.clamp(0, 1).float()
        
        hr_tensor = (imgs_hr + 1) / 2
        hr_tensor = hr_tensor.clamp(0, 1).float()

        # Metrics Calculation (Skip if enable_metrics is False)
        if enable_metrics and metrics_registry is not None:
            sr_for_metric = sr_tensor.detach()
            hr_for_metric = hr_tensor.detach()
            if misc.is_dist_avail_and_initialized():
                sr_for_metric = _gather_tensor(sr_for_metric)
                hr_for_metric = _gather_tensor(hr_for_metric)

            batch_scores = {}
            with torch.no_grad():
                for metric_name, metric in metrics_registry.full_ref_metrics.items():
                    batch_scores[metric_name] = metric(sr_for_metric, hr_for_metric)

                for metric_name, metric in metrics_registry.no_ref_metrics.items():
                    batch_scores[metric_name] = metric(sr_for_metric)

            for metric_name, metric_value in batch_scores.items():
                metric_logger.update(**{metric_name: metric_value.mean().item()})

        # Save individual SR/HR images
        save_sr_hr_images(sr_tensor, hr_tensor, filenames, sr_save_dir, hr_save_dir, i)
        
        # --- 3. 保存 4 张拼接对比图 ---
        if misc.get_rank() == 0:
            # 参数: SR, HR, SwinIR(当前的imgs_lr), LR_Raw
            save_comparison_images(sampled_images, imgs_hr, imgs_lr, imgs_lr_raw, save_folder, i)

        # Print Summary at the end
        if misc.get_rank() == 0 and total_images_processed > 0 and i == len(data_loader) - 1:
             avg_steps = total_steps_accum / total_images_processed
             avg_time = total_inference_time / total_images_processed
             
             print("\n" + "="*50)
             print(f"推理性能总结:")
             print(f"  - 平均推理步数: {avg_steps:.2f}")
             print(f"  - 处理图片总数: {total_images_processed}")
             print(f"  - 推理总耗时: {total_inference_time:.2f}s")
             print(f"  - 全局平均耗时 (每张): {avg_time:.4f}s")
             print("="*50 + "\n")

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # FID Calculation (Only if enable_metrics is True)
    if enable_metrics and metrics_registry is not None:
        fid_score = metrics_registry.fid_metric(sr_save_dir, hr_save_dir)
        fid_score_value = fid_score.item() if torch.is_tensor(fid_score) else float(fid_score)
        metric_logger.update(fid=fid_score_value)
    
    time.sleep(1)

    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    metric_logger.synchronize_between_processes()
    
    print("Averaged Validation stats:", metric_logger)

    # Write Metrics to file (Only if enable_metrics is True)
    if enable_metrics and misc.get_rank() == 0:
        txt_path = os.path.join(save_folder, "metrics_results.txt")
        with open(txt_path, "a") as f:
            log_str = f"Epoch {epoch}: {metric_logger}"          
            f.write(log_str + "\n")
        print(f"Metrics appended to: {txt_path}")
    
    # Write to Tensorboard (Only if enable_metrics is True)
    if enable_metrics and log_writer is not None:
        for name, meter in metric_logger.meters.items():
            log_writer.add_scalar(f'val/{name}', meter.global_avg, epoch)

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if prev_training_mode:
        model_without_ddp.train()

    return results

def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, moments=moments[i].cpu().numpy(), moments_flip=moments_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return
def save_sr_hr_images(sr, hr, filenames, sr_dir, hr_dir, batch_idx):
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)

    def to_numpy(x):
        #x = (x + 1) / 2
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x = x.clamp(0, 1)
        return (x * 255).byte().cpu().permute(0, 2, 3, 1).numpy()

    sr_np = to_numpy(sr)
    hr_np = to_numpy(hr)

    rank = misc.get_rank()

    for idx in range(sr_np.shape[0]):
        base_name = None
        if filenames is not None and len(filenames) > idx:
            base_name = os.path.splitext(os.path.basename(str(filenames[idx])))[0]
        if not base_name:
            base_name = f"batch{batch_idx}_img{idx}"

        prefix = f"{base_name}"
        sr_path = os.path.join(sr_dir, f"{prefix}.png")
        hr_path = os.path.join(hr_dir, f"{prefix}.png")

        sr_img = cv2.cvtColor(sr_np[idx], cv2.COLOR_RGB2BGR)
        hr_img = cv2.cvtColor(hr_np[idx], cv2.COLOR_RGB2BGR)

        cv2.imwrite(sr_path, sr_img)
        cv2.imwrite(hr_path, hr_img)
def save_comparison_images(sr, hr, swinir, lr_raw, save_folder, batch_idx):
    # 辅助函数：把 Tensor 转为 numpy 图片
    def process(x):
        # 诊断
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"\n！警告：生成结果检测到 NaN/Inf (Batch {batch_idx})")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 反归一化 & 格式转换
        x = (x + 1) / 2
        x = x.clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
        return x

    sr_np = process(sr)
    hr_np = process(hr)
    
    # 处理 SwinIR 结果 (中间图)
    # 如果尺寸不一样 (比如 SwinIR 没做上采样)，这里强制缩放到 SR 尺寸以便拼接
    if swinir.shape[-2:] != sr.shape[-2:]:
        swinir_upscaled = F.interpolate(swinir, size=sr.shape[-2:], mode='nearest')
        swinir_np = process(swinir_upscaled)
    else:
        swinir_np = process(swinir)

    # 处理 LR Raw (最左图)
    # 肯定需要放大到 SR 尺寸才能拼接
    lr_raw_upscaled = F.interpolate(lr_raw, size=sr.shape[-2:], mode='nearest')
    lr_raw_np = process(lr_raw_upscaled)

    # 遍历 Batch 拼接并保存
    # 顺序：LR原图 -> SwinIR结果 -> 模型输出 -> HR原图
    for j in range(sr_np.shape[0]):
        combined = np.concatenate([lr_raw_np[j], swinir_np[j], sr_np[j], hr_np[j]], axis=1)
        
        combined = (combined * 255).astype(np.uint8)
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        
        filename = os.path.join(save_folder, f"batch{batch_idx}_img{j}_comparison.png")
        cv2.imwrite(filename, combined)
