import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder
from dataset.dataset_sr import SRDataset
from models.vae import AutoencoderKL
from models import mar
from engine_mar import train_one_epoch, evaluate
from models.swinir import SwinIR
import copy


def get_args_parser():
    parser = argparse.ArgumentParser('MAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--degradation', default='codeformer', type=str,
                    help='Degradation type: codeformer or realesrgan')
    parser.add_argument('--use_swinir', action='store_true', help='Clean CodeFormer LR images with SwinIR')
    parser.add_argument('--swinir_ckpt', default='pretrained_models/swinir/face_full_v1.ckpt', type=str,
                        help='Path to SwinIR checkpoint used for LR preprocessing')
    parser.add_argument('--swinir_batch', type=int, default=4, help='Mini-batch size for SwinIR inference')
    # Model parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL',
                        help='Name of model to train')
    # 多尺度训练开关
    parser.add_argument('--multi_scale', action='store_true', help='Enable continuous multi-scale training')

    # VAE parameters
    parser.add_argument('--img_size', default=512, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=70000, type=int,
                        help='number of images to generate')
    parser.add_argument('--use_dynamic_maskgit', action='store_true',
                        help='Use dynamic MaskGIT with variance confidence (inference only)')
    parser.add_argument('--conf_method', type=str, default='stats', 
                    choices=['entropy', 'stats', 'cosine', 'semantic'],
                    help='Confidence method: entropy, stats, cosine, or semantic')
    parser.add_argument('--conf_threshold', type=float, default=0.0,
                        help='Confidence threshold tau for accepting tokens')
    parser.add_argument('--conf_pmin', type=float, default=0.01,
                        help='Minimum fraction of tokens to force-accept each round')
    parser.add_argument('--conf_window', type=str, default='40:10',
                        help='Confidence window steps. Formats: '
                             '1) "Start:Step" (e.g., "40:10" -> 40,30,20,10,0); '
                             '2) "Start:End:Step" (e.g., "30:20:1" -> 30..20); '
                             '3) comma-separated list.')
    #parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    # MAR params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)
    
    parser.add_argument('--use_lr_inject', action='store_true',
                        help='Enable RestoreVAR-style LR injection')
    parser.add_argument('--lr_inject_layers', default='all',
                        choices=['all', 'first_half', 'last_half', 'custom'],
                        help='LR injection layer selection')
    parser.add_argument('--lr_inject_cond_source', default='encoder',
                        choices=['encoder', 'vae_latent', 'patch_embed'],
                        help='LR condition feature source')
    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="ddim100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')
    
    parser.add_argument('--use_rope', action='store_true',
                        help='Use 2D RoPE; default is absolute positional embeddings')
    parser.add_argument('--use_mse_loss', action='store_true',
                        help='Enable MSE loss component')
    parser.add_argument('--mse_weight', default=0.2, type=float, help='Weight for MSE loss')
    # Dataset parameters
    parser.add_argument('--hr_data_path', default=None, type=str,
                        help='dataset path for High Resolution images')
    parser.add_argument('--lr_data_path', default=None, type=str,
                    help='dataset path for Low Resolution images (for paired testing)')
    parser.add_argument('--paired_test', action='store_true',
                    help='Use paired HR/LR dataset for evaluation')
    parser.add_argument('--val_data_path', default=None, type=str,
                        help='dataset path for validation (optional). If None, use hr_data_path/val')
    parser.add_argument('--steps_per_epoch', default=-1, type=int,
                        help='max steps per epoch (force stop), -1 means run full epoch')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # [修正] 提前初始化 Log Writer，确保全局可用
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
 
    # 1. 初始化变量
    dataset_train = None
    data_loader_train = None
    sampler_train = None
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # 2. 训练集加载 (仅在非评估模式下执行)
    if not args.evaluate:
        if args.use_cached:
            raise NotImplementedError("Cached mode needs update for SRDataset")
        else:
            print(f"Loading Training Dataset from {args.hr_data_path}")
            dataset_train = SRDataset(
                root=args.hr_data_path, 
                hr_size=args.img_size,
                lr_size=args.img_size // 2,
                is_train=True, 
                degradation_type=args.degradation
            )
            print(dataset_train)

            # 初始化 Sampler
            if args.distributed:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
            print("Sampler_train = %s" % str(sampler_train))

            # 初始化 DataLoader
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )

    # 3. 验证集/测试集加载 (核心修改部分)
    # 这里的 import 放在这里是为了防止在没有该文件时影响训练
    from dataset.dataset_paired import PairedSRDataset, HROnlyDataset 

    if getattr(args, 'paired_test', False) and getattr(args, 'lr_data_path', None) is not None:
        # [模式 A] 成对测试 (HR + LR)
        print(f"Loading Paired Test Dataset from {args.hr_data_path} and {args.lr_data_path}")
        dataset_val = PairedSRDataset(
            root_hr=args.hr_data_path, 
            root_lr=args.lr_data_path, 
            img_size=args.img_size
        )
    
    elif args.evaluate:
        # [模式 B] 纯 HR 测试 (自动退化，读取扁平文件夹)
        # 这就是解决你 FileNotFoundError 的关键
        print(f"Loading HR-Only Test Dataset from {args.hr_data_path} (Flat Folder)")
        dataset_val = HROnlyDataset(
            root_hr=args.hr_data_path,
            img_size=args.img_size
        )
        
    else:
        # [模式 C] 训练时的验证 (需要标准 ImageFolder 结构)
        if args.val_data_path is not None:
            val_root = args.val_data_path
        else:
            val_root = args.hr_data_path 

        dataset_val = SRDataset(
            root=val_root,
            hr_size=args.img_size,
            lr_size=args.img_size // 2,
            is_train=False
        )

    # 验证集 Sampler
    if args.distributed:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by num_tasks.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 验证集 DataLoader
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.eval_bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
# define the vae and mar model
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    vae_param_stats = misc.count_parameters(vae)
    print("VAE total parameters: {:.2f}M".format(vae_param_stats["total"] / 1e6))
    print("VAE trainable parameters: {:.2f}M".format(vae_param_stats["trainable"] / 1e6))
    if log_writer is not None:
        log_writer.add_scalar('params/vae_total_millions', vae_param_stats["total"] / 1e6, 0)
        log_writer.add_scalar('params/vae_trainable_millions', vae_param_stats["trainable"] / 1e6, 0)
    for param in vae.parameters():
        param.requires_grad = False

    swinir_model = None
    if args.use_swinir:
        print("Initializing SwinIR for LR preprocessing...")
        swinir_model = SwinIR(
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=180,
            depths=[6, 6, 6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
            window_size=8,
            mlp_ratio=2.0,
            sf=8,
            img_range=1.0,
            upsampler="nearest+conv",
            resi_connection="1conv",
            unshuffle=True,
            unshuffle_scale=8,
        )

        if os.path.exists(args.swinir_ckpt):
            print(f"Loading SwinIR weights from: {args.swinir_ckpt}")
            checkpoint = torch.load(args.swinir_ckpt, map_location='cpu')

            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            elif 'params_ema' in checkpoint:
                pretrained_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                pretrained_dict = checkpoint['params']
            else:
                pretrained_dict = checkpoint

            model_dict = swinir_model.state_dict()
            valid_dict = {}

            # === 核心修复：自动剥离 'module.' 前缀 ===
            for k, v in pretrained_dict.items():
                new_k = k
                if k.startswith('module.'):
                    new_k = k[7:]  # 去掉开头的 "module."
                
                if new_k in model_dict:
                    if v.shape == model_dict[new_k].shape:
                        valid_dict[new_k] = v
                    else:
                        print(f"Shape mismatch: {new_k} (ckpt: {v.shape} vs model: {model_dict[new_k].shape})")
            # ========================================

            swinir_model.load_state_dict(valid_dict, strict=True) # 这里可以放心开 strict=True 了，或者保持 False
            print(f"Successfully loaded {len(valid_dict)} / {len(model_dict)} keys for SwinIR.")
            
            if len(valid_dict) == 0:
                raise RuntimeError("Error: Loaded 0 keys! Checkpoint matching failed.")
        else:
            print(f"Warning: SwinIR weight not found at {args.swinir_ckpt}. Using random init (NOT RECOMMENDED).")


        swinir_model.eval()
        swinir_model.to(device)
        for param in swinir_model.parameters():
            param.requires_grad = False
    
    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
        mse_weight=args.mse_weight,
        use_lr_inject=args.use_lr_inject,
        lr_inject_layers=args.lr_inject_layers,
        lr_inject_cond_source=args.lr_inject_cond_source,
        use_rope=args.use_rope,
        use_mse_loss=args.use_mse_loss,
        use_dynamic_maskgit=args.use_dynamic_maskgit,
        conf_method=args.conf_method,
        conf_threshold=args.conf_threshold,
        conf_pmin=args.conf_pmin,
        conf_window=args.conf_window,
    )

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    param_stats = misc.count_parameters(model)
    print("Total parameters: {:.2f}M".format(param_stats["total"] / 1e6))
    print("Number of trainable parameters: {:.2f}M".format(param_stats["trainable"] / 1e6))
    if log_writer is not None:
        log_writer.add_scalar('params/total_millions', param_stats["total"] / 1e6, 0)
        log_writer.add_scalar('params/trainable_millions', param_stats["trainable"] / 1e6, 0)

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # no weight decay on bias, norm layers, and diffloss MLP
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # resume training
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu', weights_only=False)
        missing, unexpected = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if missing or unexpected:
            print(f"Checkpoint loaded with missing keys: {missing} and unexpected keys: {unexpected}")
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not args.evaluate:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 use_ema=True, data_loader=data_loader_val, paired_mode=args.paired_test,
                 swinir_model=swinir_model)
        return

    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            swinir_model=swinir_model
        )

        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")

        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                     use_ema=True, data_loader=data_loader_val
                     , swinir_model=swinir_model
                     )
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
