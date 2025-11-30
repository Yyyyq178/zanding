#封装了具体的训练和评估逻辑。
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
from models.vae import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import copy
import time


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

    for data_iter_step, (samples_hr, samples_lr) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # 如果当前步数达到了我们设定的限制，直接强行结束这一轮，进入 Validation
        if args.steps_per_epoch > 0 and data_iter_step >= args.steps_per_epoch:
            break

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_steps_per_epoch + epoch, args)

        # 把数据移到 GPU
        samples_hr = samples_hr.to(device, non_blocking=True)
        samples_lr = samples_lr.to(device, non_blocking=True)
        #samples_lr = samples_lr.to(device, non_blocking=True)

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
        #with torch.amp.autocast('cuda'):
            # 传入模型
        loss = model(x_hr, x_lr)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

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
             use_ema=True, data_loader=None):
    torch.cuda.empty_cache()
    model_without_ddp.eval()
    #num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
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
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

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

    print(f"Start evaluation on {len(data_loader)} batches...")
    
    # 开始遍历验证集 (HR, LR)
    for i, (imgs_hr, imgs_lr) in enumerate(data_loader):
        
        # 1. 准备数据
        imgs_hr = imgs_hr.cuda(non_blocking=True)
        #imgs_lr = imgs_lr.cuda(non_blocking=True)
        imgs_lr = imgs_lr.cuda(non_blocking=True)
        # 为了节省时间，只跑前 5 个 batch 看效果
        if i >= 5: 
            print("Finished 5 batches preview, stopping evaluation.")
            break 
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
            #with torch.amp.autocast('cuda'):
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

        # 4. 保存图片 (调用辅助函数)
        if misc.get_rank() == 0: # 只在主进程保存
            save_comparison_images(sampled_images, imgs_hr, imgs_lr, save_folder, i)
    torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    # if log_writer is not None:
    #     if args.img_size == 256:
    #         input2 = None
    #         fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
    #     else:
    #         raise NotImplementedError
    #     metrics_dict = torch_fidelity.calculate_metrics(
    #         input1=save_folder,
    #         input2=input2,
    #         fid_statistics_file=fid_statistics_file,
    #         cuda=True,
    #         isc=True,
    #         fid=True,
    #         kid=False,
    #         prc=False,
    #         verbose=False,
    #     )
    #     fid = metrics_dict['frechet_inception_distance']
    #     inception_score = metrics_dict['inception_score_mean']
    #     postfix = ""
    #     if use_ema:
    #        postfix = postfix + "_ema"
    #     if not cfg == 1.0:
    #        postfix = postfix + "_cfg{}".format(cfg)
    #     log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
    #     log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
    #     print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
    #     # remove temporal saving folder
    #     shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)


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
        # --- 🟢【新增】第一步：诊断是否炸了 ---
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"\n⚠️ 警告：在生成结果中检测到 NaN 或 Inf！(Batch {batch_idx})")
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