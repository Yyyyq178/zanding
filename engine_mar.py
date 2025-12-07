#封装了具体的训练和评估逻辑。
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
from models.vae import DiagonalGaussianDistribution
from dataset.codeformer import CodeFormerDegradation
import pyiqa
import shutil
import cv2
import numpy as np
import os
import copy
import time

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
        

    for data_iter_step, (samples_hr, samples_lr, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # 如果当前步数达到了我们设定的限制，直接强行结束这一轮，进入 Validation
        if args.steps_per_epoch > 0 and data_iter_step >= args.steps_per_epoch:
            break

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_steps_per_epoch + epoch, args)

        # 把数据移到 GPU
        samples_hr = samples_hr.to(device, non_blocking=True)

        # if args.multi_scale: 
            # scale=None -> 随机 2-4 倍
        samples_lr = degradation_model(samples_hr, scale=None) 
        # else:
        #     # 固定 4 倍
        #     samples_lr = degradation_model(samples_hr, scale=4.0)

        # # ================= SwinIR 预处理(一次一张) =================
        # if swinir_model is not None:
        #     with torch.no_grad():
        #         # 范围转换 [-1, 1] -> [0, 1]
        #         lr_input = (samples_lr + 1) / 2
        #         lr_cleaned_list = []
        #         mini_batch_size = 1
        #         for i in range(0, lr_input.size(0), mini_batch_size):
        #             # 取出切片
        #             mini_input = lr_input[i : i + mini_batch_size]
        #             # 推理
        #             mini_output = swinir_model(mini_input)
        #             lr_cleaned_list.append(mini_output)
                
        #         # 拼回完整的 Batch
        #         lr_cleaned = torch.cat(lr_cleaned_list, dim=0)
        #         # -----------------------------------

        #         # 范围转换回 [-1, 1]
        #         samples_lr = (lr_cleaned * 2) - 1
                
        #         # [修复报错] 显存清理 & 删除不存在的变量
        #         del lr_input, lr_cleaned, lr_cleaned_list, mini_input, mini_output
        # # ===========================================================
        # ================= SwinIR 预处理 =================
        # if swinir_model is not None:
        #     with torch.no_grad():
        #         # 范围转换 [-1, 1] -> [0, 1]
        #         lr_input = (samples_lr + 1) / 2
                
        #         # === [修改回退] 直接批量推理 ===
        #         # 既然分辨率已经改回 512，这里直接送入整个 batch
        #         lr_cleaned = swinir_model(lr_input)
        #         # 范围转换回 [-1, 1]
        #         samples_lr = (lr_cleaned * 2) - 1
                
        #         del lr_input, lr_cleaned
        # # ===========================================================

        with torch.no_grad():
            if args.use_cached:
                 raise NotImplementedError("Cached mode not supported for SR yet.")
            else:
                # === 编码 HR ===
                posterior_hr = vae.encode(samples_hr)
                x_hr = posterior_hr.sample().mul_(0.2325) 

                # === 编码 LR ===
                posterior_lr = vae.encode(samples_lr)
                x_lr = posterior_lr.sample().mul_(0.2325)

        # forward
        with torch.amp.autocast('cuda'):
            # 传入模型
            loss = model(x_hr, x_lr)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        #torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True, data_loader=None, swinir_model=None):
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
    print("Loading metrics models (PSNR/SSIM/LPIPS)...")
    # 这里的 device='cuda' 假设你一定用 GPU。LPIPS 计算较慢，如果不关心可以注释掉。
    psnr_metric = pyiqa.create_metric('psnr', device='cuda')
    ssim_metric = pyiqa.create_metric('ssim', device='cuda')
    lpips_metric = pyiqa.create_metric('lpips', device='cuda') 
    
    # 使用 misc.MetricLogger 来自动管理所有 GPU 上的平均分计算
    metric_logger = misc.MetricLogger(delimiter="  ")
    # -------------------------------------------
    degradation_model = CodeFormerDegradation()
    print(f"Start evaluation on {len(data_loader)} batches...")
    
    # 开始遍历验证集 (HR, LR)
    for i, (imgs_hr, imgs_lr, filenames) in enumerate(data_loader):
        
        # 1. 准备数据
        imgs_hr = imgs_hr.cuda(args.device, non_blocking=True)
        #imgs_lr = imgs_lr.cuda(non_blocking=True)
        #imgs_lr = imgs_lr.cuda(args.device, non_blocking=True)
        # 为了节省时间，只跑前 5 个 batch 看效果
        if i >= 5: 
            print("Finished 5 batches preview, stopping evaluation.")
            break 

        imgs_lr = degradation_model(imgs_hr, scale=4.0)
        
        # # SwinIR 预处理（一次一张图片）
        # if swinir_model is not None:
        #     with torch.no_grad():
        #         lr_input = (imgs_lr + 1) / 2
                
        #         # --- [显存优化] 验证集切片推理 ---
        #         lr_cleaned_list = []
        #         mini_batch_size = 1
        #         for j in range(0, lr_input.size(0), mini_batch_size):
        #             mini_input = lr_input[j : j + mini_batch_size]
        #             mini_output = swinir_model(mini_input)
        #             lr_cleaned_list.append(mini_output)
        #         lr_cleaned = torch.cat(lr_cleaned_list, dim=0)
        #         # -------------------------------

        #         imgs_lr = (lr_cleaned * 2) - 1
        #         del lr_input, lr_cleaned, lr_cleaned_list

        # # SwinIR 预处理
        # if swinir_model is not None:
        #     with torch.no_grad():
        #         lr_input = (imgs_lr + 1) / 2
                
        #         # === 直接批量推理 ===
        #         lr_cleaned = swinir_model(lr_input)
                
        #         samples_lr = (lr_cleaned * 2) - 1
                
        #         # 变量名对齐
        #         imgs_lr = samples_lr 
        #         del lr_input, lr_cleaned

        # 2. 编码 LR (作为条件)
        with torch.no_grad():
            posterior_lr = vae.encode(imgs_lr)
            x_lr = posterior_lr.sample().mul_(0.2325) 

        # 计算目标 Token 数量 (HR Latent 的长度)
        h_hr, w_hr = imgs_hr.shape[-2:]
        feat_h, feat_w = h_hr // 16, w_hr // 16 # 假设 stride=16
        target_seq_len = feat_h * feat_w

        # 3. 生成 (Inference)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # 调用 sample_tokens，传入 x_lr
                sampled_tokens = model_without_ddp.sample_tokens(
                    bsz=imgs_lr.shape[0], 
                    num_iter=args.num_iter, 
                    cfg=cfg,
                    cfg_schedule=args.cfg_schedule, 
                    x_lr=x_lr,
                    temperature=args.temperature,
                    target_seq_len=target_seq_len  
                )
            
                # 解码生成的 Token 变回图片
                sampled_images = vae.decode(sampled_tokens / 0.2325)

        # 1. 数据预处理：将范围从 [-1, 1] 转换到 [0, 1]
        # 注意：pyiqa 期望输入是 [0, 1] 的 float32 Tensor
        sr_tensor = (sampled_images + 1) / 2
        sr_tensor = sr_tensor.clamp(0, 1).float()  # 截断防止越界
        
        hr_tensor = (imgs_hr + 1) / 2
        hr_tensor = hr_tensor.clamp(0, 1).float()

        # 2. 计算当前 batch 的分数
        with torch.no_grad():
            # pyiqa 会返回这个 batch 的平均分 (scalar tensor)
            batch_psnr = psnr_metric(sr_tensor, hr_tensor)
            batch_ssim = ssim_metric(sr_tensor, hr_tensor)
            batch_lpips = lpips_metric(sr_tensor, hr_tensor)

        # 3. 记录分数 (MetricLogger 会自动处理累加和平滑)
        metric_logger.update(psnr=batch_psnr.mean().item())
        metric_logger.update(ssim=batch_ssim.mean().item())
        metric_logger.update(lpips=batch_lpips.mean().item())
        # 4. 保存图片 (调用辅助函数)
        if misc.get_rank() == 0: # 只在主进程保存
            save_comparison_images(sampled_images, imgs_hr, imgs_lr, save_folder, i)

    torch.distributed.barrier()
    time.sleep(1)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # 1. 同步所有显卡的统计数据 (这一步非常重要，否则你看到的只有主卡的分数)
    metric_logger.synchronize_between_processes()
    
    print("Averaged Validation stats:", metric_logger)

    # 将指标保存到 TXT 文件
    if misc.get_rank() == 0:
        txt_path = os.path.join(save_folder, "metrics_results.txt")
        
        with open(txt_path, "a") as f:
            # 构造单行日志字符串
            # str(metric_logger) 会自动生成 "psnr: X (X)  ssim: X (X) ..." 的格式
            log_str = f"Epoch {epoch}: {metric_logger}"
            
            # 如果计算了 FID，把它拼接到行尾
            # if 'fid_score' in locals() and fid_score > 0:
            #     log_str += f"  fid: {fid_score:.4f}"
            
            f.write(log_str + "\n")
            
        print(f"Metrics appended to: {txt_path}")
    
    # 2. 如果配置了 Tensorboard，写入验证集指标
    if log_writer is not None:
        # 注意：这里假设你在第二步中已经像 metric_logger.update(psnr=...) 那样添加了这些 key
        if 'psnr' in metric_logger.meters:
            log_writer.add_scalar('val/psnr', metric_logger.meters['psnr'].global_avg, epoch)
        if 'ssim' in metric_logger.meters:
            log_writer.add_scalar('val/ssim', metric_logger.meters['ssim'].global_avg, epoch)
        if 'lpips' in metric_logger.meters:    
            log_writer.add_scalar('val/lpips', metric_logger.meters['lpips'].global_avg, epoch)
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
