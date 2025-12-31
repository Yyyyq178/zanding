import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
from models.vae import DiagonalGaussianDistribution
from dataset.codeformer import CodeFormerDegradation
from util.deg_utils import build_degradation_map, pool_degradation_to_tokens
import pyiqa
import shutil
import cv2
import numpy as np
import os
import copy
import time

#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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

    degradation_model = CodeFormerDegradation()

    gate_multiplier = 1.0
    num_iter = args.decode_steps if (args.curriculum_decode and args.decode_steps is not None) else args.num_iter
    global_step_offset = epoch * num_steps_per_epoch

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

        d_tok_gt = None
        if args.use_deg_head:
            d_pix_gt = build_degradation_map(samples_hr, samples_lr, args.deg_w_pix, args.deg_w_grad)
            d_tok_gt = pool_degradation_to_tokens(d_pix_gt, model)

        with torch.amp.autocast('cuda'):
            model_out = model(
                x_hr, x_lr, gate_multiplier=gate_multiplier,
                curriculum_decode=args.curriculum_decode,
                num_iter=num_iter,
                d_tok_gt=d_tok_gt,
                global_step=global_step_offset + data_iter_step,
                curriculum_pred_order_warmup_steps=args.curriculum_pred_order_warmup_steps,
                curriculum_pred_order_prob_max=args.curriculum_pred_order_prob_max,
            )

            if args.use_deg_head:
                loss, loss_diff, loss_mse, d_tok_pred = model_out
            else:
                loss, loss_diff, loss_mse = model_out
                d_tok_pred = None

            if args.use_deg_head:
                assert d_tok_pred is not None
                d_tok_gt = d_tok_gt.to(d_tok_pred.device)
                assert d_tok_pred.shape == d_tok_gt.shape
                loss_deg = F.l1_loss(d_tok_pred, d_tok_gt)
                loss = loss + args.lambda_deg * loss_deg

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

        if args.use_deg_head:
            loss_deg_value = loss_deg.item()
            d_mean = d_tok_pred.mean().item()
            metric_logger.update(loss_deg=loss_deg_value)
            metric_logger.update(d_tok_mean=d_mean)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_diff', misc.all_reduce_mean(loss_diff.item()), epoch_1000x)
            log_writer.add_scalar('train_loss_mse', misc.all_reduce_mean(loss_mse.item()), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if args.use_lr_inject:
                log_writer.add_scalar('train_lr_inject_gate_mean', misc.all_reduce_mean(gate_mean), epoch_1000x)
                log_writer.add_scalar('train_lr_inject_gate_multiplier', misc.all_reduce_mean(gate_multiplier), epoch_1000x)
            if args.use_deg_head:
                log_writer.add_scalar('train_loss_deg', misc.all_reduce_mean(loss_deg_value), epoch_1000x)
                log_writer.add_scalar('train_d_tok_mean', misc.all_reduce_mean(d_mean), epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True, data_loader=None, swinir_model=None, paired_mode=False):
    torch.cuda.empty_cache()
    model_without_ddp.eval()
    save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}cfg{}-image_num{}".format(args.num_iter,
                                                                                                     args.num_sampling_steps,
                                                                                                     args.temperature,
                                                                                                     args.cfg_schedule,
                                                                                                     cfg,
                                                                                                     args.num_images))
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)

    if misc.get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    used_time = 0

    gen_img_cnt = 0

    if data_loader is None:
        print("No data loader provided for evaluation, skipping.")
        return
    
    # --- 初始化评价指标 (加载模型到 GPU) ---
    print("Loading metrics models (PSNR/SSIM/LPIPS/FID/MUSIQ/BRISQUE/DISTS/MANIQA/CLIPIQA/MS-SSIM/NIQE/PIQE)...")
    # 这里的 device='cuda' 假设你一定用 GPU。LPIPS 计算较慢，如果不关心可以注释掉。
    metric_specs = {
        # 全参考 (需要 hr_tensor)
        'psnr': ('psnr', True),
        'ssim': ('ssim', True),
        'lpips': ('lpips', True),
        'dists': ('dists', True),
        'ms_ssim': ('ms_ssim', True),
        'musiq': ('musiq', False),
        'brisque': ('brisque', False),
        'maniqa': ('maniqa', False),
        'clipiqa': ('clipiqa', False),
        'niqe': ('niqe', False),
        'piqe': ('piqe', False),
    }

    fid_metric = pyiqa.create_metric('fid', device='cuda')

    full_ref_metrics = {}
    no_ref_metrics = {}

    for log_name, (metric_name, is_full_ref) in metric_specs.items():
        metric = pyiqa.create_metric(metric_name, device='cuda')
        if is_full_ref:
            full_ref_metrics[log_name] = metric
        else:
            no_ref_metrics[log_name] = metric
    
    # 使用 misc.MetricLogger 来自动管理所有 GPU 上的平均分计算
    metric_logger = misc.MetricLogger(delimiter="  ")
    # -------------------------------------------
    if not paired_mode:
        degradation_model = CodeFormerDegradation()

    sr_save_dir = os.path.join(save_folder, "sr_images")
    hr_save_dir = os.path.join(save_folder, "hr_images")

    print(f"Start evaluation on {len(data_loader)} batches...")
    
    # 开始遍历验证集 (HR, LR)
    for i, (imgs_hr, imgs_lr, filenames) in enumerate(data_loader):

        # 准备数据
        imgs_hr = imgs_hr.cuda(args.device, non_blocking=True)
        imgs_lr = imgs_lr.cuda(args.device, non_blocking=True)

        # 为了节省时间，只跑前 5 个 batch 看效果
        if i >= 5: 
            print("Finished 5 batches preview, stopping evaluation.")
            break 
        if not paired_mode:
            # 如果是单独运行测试脚本 (args.evaluate=True)，且不是在线验证 (args.online_eval=False)
            # 或者你可以直接简单粗暴地判断：如果是测试模式，就用随机
            if args.evaluate:
                # 测试模式：传入 None，激活 CodeFormer 内部的 (1, 12) 随机逻辑
                imgs_lr = degradation_model(imgs_hr, scale=None)
            else:
                # 训练验证模式：固定 4.0，保证指标稳定
                imgs_lr = degradation_model(imgs_hr, scale=4.0)
        
        if swinir_model is not None:
            imgs_lr = preprocess_with_swinir(imgs_lr, swinir_model, args.swinir_batch)

        # 编码 LR (作为条件)
        with torch.no_grad():
            posterior_lr = vae.encode(imgs_lr)
            x_lr = posterior_lr.sample().mul_(0.2325) 

        # 计算目标 Token 数量 (HR Latent 的长度)
        h_hr, w_hr = imgs_hr.shape[-2:]
        feat_h, feat_w = h_hr // 16, w_hr // 16 # 假设 stride=16
        target_seq_len = feat_h * feat_w

        # 生成 (Inference)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # 调用 sample_tokens，传入 x_lr
                num_iter = args.decode_steps if (args.curriculum_decode and args.decode_steps is not None) else args.num_iter
                sampled_tokens = model_without_ddp.sample_tokens(
                    bsz=imgs_lr.shape[0], 
                    num_iter=num_iter, 
                    cfg=cfg,
                    cfg_schedule=args.cfg_schedule, 
                    x_lr=x_lr,
                    temperature=args.temperature,
                    target_seq_len=target_seq_len,
                    curriculum_decode=args.curriculum_decode,
                )
            
                # 解码生成的 Token 变回图片
                sampled_images = vae.decode(sampled_tokens / 0.2325)

        # 数据预处理：将范围从 [-1, 1] 转换到 [0, 1]
        # 注意：pyiqa 期望输入是 [0, 1] 的 float32 Tensor
        sr_tensor = (sampled_images + 1) / 2
        sr_tensor = sr_tensor.clamp(0, 1).float()  # 截断防止越界
        
        hr_tensor = (imgs_hr + 1) / 2
        hr_tensor = hr_tensor.clamp(0, 1).float()

        # 计算当前 batch 的分数
        batch_scores = {}
        with torch.no_grad():
            # pyiqa 会返回这个 batch 的平均分 (scalar tensor)
            for metric_name, metric in full_ref_metrics.items():
                batch_scores[metric_name] = metric(sr_tensor, hr_tensor)

            for metric_name, metric in no_ref_metrics.items():
                batch_scores[metric_name] = metric(sr_tensor)

        # 记录分数 (MetricLogger 会自动处理累加和平滑)
        for metric_name, metric_value in batch_scores.items():
            metric_logger.update(**{metric_name: metric_value.mean().item()})

        # 保存 SR/HR 图片用于后续 FID 计算
        save_sr_hr_images(sr_tensor, hr_tensor, filenames, sr_save_dir, hr_save_dir, i)

        # 保存图片 (调用辅助函数)
        if misc.get_rank() == 0:
            save_comparison_images(sampled_images, imgs_hr, imgs_lr, save_folder, i)

    torch.distributed.barrier()

    fid_score = fid_metric(sr_save_dir, hr_save_dir)
    fid_score_value = fid_score.item() if torch.is_tensor(fid_score) else float(fid_score)
    fid_score_value = misc.all_reduce_mean(torch.tensor(fid_score_value, device=args.device)).item()
    metric_logger.update(fid=fid_score_value)
    time.sleep(1)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # 同步所有显卡的统计数据 (这一步非常重要，否则你看到的只有主卡的分数)
    metric_logger.synchronize_between_processes()
    
    print("Averaged Validation stats:", metric_logger)

    # 将指标保存到 TXT 文件
    if misc.get_rank() == 0:
        txt_path = os.path.join(save_folder, "metrics_results.txt")
        
        with open(txt_path, "a") as f:
            log_str = f"Epoch {epoch}: {metric_logger}"          
            f.write(log_str + "\n")
            
        print(f"Metrics appended to: {txt_path}")
    
    # 2. 如果配置了 Tensorboard，写入验证集指标
    if log_writer is not None:
        # 注意：这里假设你在第二步中已经像 metric_logger.update(psnr=...) 那样添加了这些 key
        for name, meter in metric_logger.meters.items():
            log_writer.add_scalar(f'val/{name}', meter.global_avg, epoch)
    # 返回统计结果，方便外部调用
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    #torch.distributed.barrier()
    time.sleep(1)


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

        prefix = f"rank{rank}_{base_name}"
        sr_path = os.path.join(sr_dir, f"{prefix}.png")
        hr_path = os.path.join(hr_dir, f"{prefix}.png")

        sr_img = cv2.cvtColor(sr_np[idx], cv2.COLOR_RGB2BGR)
        hr_img = cv2.cvtColor(hr_np[idx], cv2.COLOR_RGB2BGR)

        cv2.imwrite(sr_path, sr_img)
        cv2.imwrite(hr_path, hr_img)
def save_comparison_images(sr, hr, lr, save_folder, batch_idx):
    # 辅助函数：把 Tensor 转为 numpy 图片
    def process(x):
        # --- 第一步：诊断是否炸了 ---
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"\n！警告：在生成结果中检测到 NaN 或 Inf！(Batch {batch_idx})")
            print(f"    统计: Min={x.min().item():.4f}, Max={x.max().item():.4f}, Mean={x.mean().item():.4f}")
            print("    -> 这说明模型训练不稳定（学习率太大），建议降低 blr。")
        
        # --- 正常处理流程 ---
        # 反归一化 (从 -1~1 映射回 0~1)
        x = (x + 1) / 2
        
        # 如果是 NaN，变成 0 (黑)；如果是 Inf，变成 1 (白)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 限制范围并转 numpy
        return x.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    
    sr_np = process(sr)
    hr_np = process(hr)
    
    # 把 LR 放大到和 SR 一样大
    lr_upscaled = F.interpolate(lr, size=sr.shape[-2:], mode='nearest')
    lr_np = process(lr_upscaled)
    
    # 遍历这个 batch 里的每张图并保存
    for j in range(sr_np.shape[0]):
        combined = np.concatenate([lr_np[j], sr_np[j], hr_np[j]], axis=1)
        combined = (combined * 255).astype(np.uint8)
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        
        filename = os.path.join(save_folder, f"batch{batch_idx}_img{j}_comparison.png")
        cv2.imwrite(filename, combined)
