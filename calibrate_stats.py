import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from models import mar 
from models.confidence import VarianceConfidenceAccumulator
from dataset.dataset_sr import SRDataset
from dataset.dataset_paired import PairedSRDataset 
from dataset.codeformer import CodeFormerDegradation 

def get_args_parser():
    parser = argparse.ArgumentParser('MAR Confidence Calibration', add_help=False)
    
    # 基础配置
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model', default='mar_base', type=str)
    parser.add_argument('--resume', default='', required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', default='./stats_output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=4, type=int)
    
    # 数据集配置
    parser.add_argument('--hr_data_path', default='', required=True, type=str)
    parser.add_argument('--lr_data_path', default=None, type=str, help='Path to LR images (Optional)')
    parser.add_argument('--img_size', default=512, type=int)
    
    # 模型参数 (需与推理/训练配置一致)
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str)
    parser.add_argument('--vae_embed_dim', default=16, type=int)
    parser.add_argument('--vae_stride', default=16, type=int)
    parser.add_argument('--patch_size', default=1, type=int)
    parser.add_argument('--diffloss_d', type=int, default=6)
    parser.add_argument('--diffloss_w', type=int, default=1024)
    parser.add_argument('--num_sampling_steps', type=str, default="ddim100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    
    # 辅助开关
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--mse_weight', default=0.2, type=float)
    parser.add_argument('--use_lr_inject', action='store_true')
    parser.add_argument('--lr_inject_layers', default='all')
    parser.add_argument('--lr_inject_cond_source', default='encoder')
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--use_mse_loss', action='store_true')
    parser.add_argument('--mask_ratio_min', type=float, default=0.7)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--proj_dropout', type=float, default=0.1)
    parser.add_argument('--buffer_size', type=int, default=64)
    
    # 校准专用参数
    parser.add_argument('--conf_window', type=str, default='40:10', help='Timesteps window for collecting stats')
    parser.add_argument('--calib_batches', default=50, type=int)
    
    # 兼容性参数 (推理脚本中存在但校准不使用的参数)
    parser.add_argument('--use_dynamic_maskgit', action='store_true')
    parser.add_argument('--conf_threshold', type=float, default=0.0)
    parser.add_argument('--conf_pmin', type=float, default=0.01)

    return parser

@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    print(f"Loading model: {args.model}")
    if args.model in mar.__dict__:
        model_func = mar.__dict__[args.model]
    else:
        raise ValueError(f"Model '{args.model}' not found in models.mar")

    # 初始化 MAR 模型
    model = model_func(
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        use_lr_inject=args.use_lr_inject,
        use_rope=args.use_rope,
        conf_window=args.conf_window
    )
    
    # 加载权重 (weights_only=False 用于兼容包含完整对象的 checkpoint)
    print(f"Loading checkpoint from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 初始化 VAE
    from models.vae import AutoencoderKL
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).to(device).eval()

    # 初始化数据集
    if args.lr_data_path:
        print(f"Using Paired Dataset.")
        dataset = PairedSRDataset(
            root_hr=args.hr_data_path,
            root_lr=args.lr_data_path,
            img_size=args.img_size
        )
        degradation_model = None 
    else:
        print(f"Using Online Degradation (CodeFormer).")
        dataset = SRDataset(
            root=args.hr_data_path,
            hr_size=args.img_size,
            lr_size=args.img_size // 2,
            is_train=False, 
            degradation_type='codeformer'
        )
        degradation_model = CodeFormerDegradation().to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    
    # 确定统计窗口
    window_steps = model._get_confidence_window()
    if not window_steps:
        window_steps = list(range(model.diffloss.gen_diffusion.num_timesteps))
    print(f"Confidence Window Steps: {window_steps}")
    print(f"Starting calibration on {args.calib_batches} batches...")
    
    all_u_values = []
    
    # 主循环
    for i, batch in enumerate(tqdm(dataloader, total=args.calib_batches)):
        if i >= args.calib_batches:
            break
            
        imgs_hr = batch[0].to(device)
        if args.lr_data_path:
            imgs_lr = batch[1].to(device)
        else:
            imgs_lr = degradation_model(imgs_hr, scale=None)
        
        # 数据有效性检查
        if torch.isnan(imgs_hr).any() or torch.isnan(imgs_lr).any():
            continue

        # VAE Encoding
        posterior_hr = vae.encode(imgs_hr)
        x_hr = posterior_hr.sample().mul_(0.2325)
        posterior_lr = vae.encode(imgs_lr)
        x_lr = posterior_lr.sample().mul_(0.2325)
        
        lr_tokens = model.patchify(x_lr)
        hr_tokens = model.patchify(x_hr)
        
        # 随机 Mask 模拟
        bsz, num_tokens, _ = hr_tokens.shape
        orders = model.sample_orders(bsz, num_tokens)
        mask = model.random_masking(hr_tokens, orders)
        
        if mask.sum() == 0:
            continue

        # 构建 Condition Tokens
        cond_tokens_encoder = None
        cond_tokens_decoder = None
        if model.use_lr_inject:
            cond_tokens_base = model._build_lr_inject_cond_tokens(x_lr, lr_tokens=lr_tokens)
            cond_tokens_encoder = model.lr_inject_cond_proj_encoder(cond_tokens_base)
            cond_tokens_decoder = model.lr_inject_cond_proj_decoder(cond_tokens_base)
            if not model.use_rope:
                lr_pos_enc = model.lr_pos_embed_encoder[:, : cond_tokens_encoder.shape[1], :]
                lr_pos_dec = model.lr_pos_embed_decoder[:, : cond_tokens_decoder.shape[1], :]
                cond_tokens_encoder = cond_tokens_encoder + lr_pos_enc
                cond_tokens_decoder = cond_tokens_decoder + lr_pos_dec
        
        shape_hr = (model.seq_h, model.seq_w)
        shape_lr = (model.seq_h, model.seq_w)

        # MAE Forward
        x = model.forward_mae_encoder(
            hr_tokens, mask, lr_tokens, shape_hr, shape_lr,
            cond_tokens=cond_tokens_encoder
        )
        z, pos_embed = model.forward_mae_decoder(
            x, mask, shape_hr, shape_lr,
            cond_tokens=cond_tokens_decoder
        )
        
        # 提取被遮挡部分
        mask_bool = mask.bool()
        indices_to_pred = mask_bool.nonzero(as_tuple=True)
        z_sub = z[indices_to_pred]
        pos_sub = pos_embed[indices_to_pred]
        z_cond = z_sub + pos_sub
        
        if torch.isnan(z_cond).any():
             continue

        # 扩散采样收集不确定度
        conf_acc = VarianceConfidenceAccumulator(window_steps, collect_conf_stats=False)
        _, u_map = model.diffloss.sample(
            z_cond, temperature=1.0, confidence_accumulator=conf_acc, return_confidence=True
        )
        
        if u_map is not None and not torch.isnan(u_map).any():
            all_u_values.append(u_map.detach().cpu().flatten())

    # 计算并保存统计量
    if len(all_u_values) > 0:
        full_u = torch.cat(all_u_values, dim=0).numpy()
        mu_u = np.mean(full_u)
        sigma_u = np.std(full_u)
        
        print(f"Calibration Result: mu_u={mu_u:.6f}, sigma_u={sigma_u:.6f}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "confidence_stats.npz")
        np.savez(save_path, mu_u=mu_u, sigma_u=sigma_u)
        print(f"Statistics saved to: {save_path}")
    else:
        print("Error: No valid data collected. Please check dataset or window settings.")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)