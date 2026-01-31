import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from models import mar 
from dataset.dataset_sr import SRDataset
from dataset.dataset_paired import PairedSRDataset 
from dataset.codeformer_face import CodeFormerDegradation as CodeFormerDegradationFace
from dataset.realesrgan_natural import RealESRGANDegradationNatural
from models.swinir import SwinIR
from engine_mar import preprocess_with_swinir

def get_args_parser():
    parser = argparse.ArgumentParser('MAR Trajectory Confidence Calibration', add_help=False)
    
    # 基础配置
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model', default='mar_base', type=str)
    parser.add_argument('--resume', default='', required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', default='./pretrained_models', help='Directory to save stats')
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
    parser.add_argument('--calib_batches', default=50, type=int, help='Number of batches to use for calibration')

    # CFG 相关参数
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='CFG scale for calibration')
    parser.add_argument('--cfg_null_mode', default='zero', choices=['zero']) 
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature') 
    
    # 兼容性参数 (推理脚本中存在但校准不使用的参数)
    parser.add_argument('--use_dynamic_maskgit', action='store_true')
    parser.add_argument('--conf_threshold', type=float, default=0.0)
    parser.add_argument('--conf_pmin', type=float, default=0.01)
    parser.add_argument('--degradation', default='codeformer', type=str,
                        help='Degradation type: codeformer (face) or realesrgan_natural')
    
    parser.add_argument('--use_swinir', action='store_true', help='Use SwinIR for preprocessing')
    parser.add_argument('--swinir_ckpt', type=str, default='pretrained_models/swinir/face_swinir_v1.ckpt')
    return parser

class DeltaCollector:
    """
    用于在扩散采样循环中捕获 x0 的变化轨迹。
    MD方案：计算窗口内相邻步 pred_xstart 的差分 delta。
    """
    def __init__(self, window_steps):
        self.window_steps = set(window_steps)
        self.x0_prev = None
        # 存储当前 batch 内收集到的所有 delta
        # List of Tensor [B, C, H, W]
        self.deltas = [] 

    def update(self, t, x_start, **kwargs):
        """
        此方法会被 diffloss.p_sample_loop_progressive 调用
        t: int, 当前时间步
        x_start: Tensor, 当前预测的 x0 (pred_xstart)
        """
        if t not in self.window_steps:
            return
        
        if self.x0_prev is not None:
            delta = x_start - self.x0_prev 
            
            self.deltas.append(delta.detach().cpu())
        
        self.x0_prev = x_start.detach().clone()
        
    def finalize(self):
        return None

@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    print(f"Loading model: {args.model}")
    if args.model in mar.__dict__:
        model_func = mar.__dict__[args.model]
    else:
        raise ValueError(f"Model '{args.model}' not found in models.mar")

    # 1. 初始化 MAR 模型
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
        conf_window=args.conf_window, # 这里会解析窗口字符串
        cfg_null_mode=args.cfg_null_mode # 传入 CFG 模式
    )
    
    # 2. 加载权重
    print(f"Loading checkpoint from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 3. 初始化 VAE
    from models.vae import AutoencoderKL
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).to(device).eval()
    swinir_model = None
    if args.use_swinir:
        print(f"Loading SwinIR model from {args.swinir_ckpt}...")
        # 硬编码配置 (Real-World SR Large)
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
        
        checkpoint_swinir = torch.load(args.swinir_ckpt, map_location='cpu')
        if 'params_ema' in checkpoint_swinir:
            state_dict_swinir = checkpoint_swinir['params_ema']
        elif 'params' in checkpoint_swinir:
            state_dict_swinir = checkpoint_swinir['params']
        elif 'state_dict' in checkpoint_swinir:
            state_dict_swinir = checkpoint_swinir['state_dict']
        else:
            state_dict_swinir = checkpoint_swinir
        
        # 去除 module. 前缀
        clean_state_dict = {}
        for k, v in state_dict_swinir.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v
                
        swinir_model.load_state_dict(clean_state_dict, strict=True)
        swinir_model.to(device)
        swinir_model.eval()
        
        # 冻结 SwinIR 参数
        for param in swinir_model.parameters():
            param.requires_grad = False

    # 4. 初始化数据集
    if args.lr_data_path:
        print(f"Using Paired Dataset: HR={args.hr_data_path}, LR={args.lr_data_path}")
        dataset = PairedSRDataset(
            root_hr=args.hr_data_path,
            root_lr=args.lr_data_path,
            img_size=args.img_size
        )
        degradation_model = None 
    else:
        print(f"Using Online Degradation (CodeFormer) on {args.hr_data_path}")
        dataset = SRDataset(
            root=args.hr_data_path,
            hr_size=args.img_size,
            lr_size=args.img_size // 2,
            is_train=False,
        )
        if args.degradation == 'realesrgan_natural':
            print("Initializing Natural Degradation (Fixed 4x)...")
            degradation_model = RealESRGANDegradationNatural().to(device)
        else:
            print("Initializing Face Degradation (Random 1-12x)...")
            degradation_model = CodeFormerDegradationFace().to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    
    # 5. 获取窗口步数
    window_steps = model._get_confidence_window()
    if not window_steps:
        # 如果未指定，默认覆盖所有步数 (通常不建议，应指定如 40:10)
        window_steps = list(range(model.diffloss.gen_diffusion.num_timesteps))
    
    print(f"Calibration Window Steps: {window_steps}")
    print(f"CFG Scale: {args.cfg_scale}, Temperature: {args.temperature}")
    if args.cfg_scale > 1.0:
        print(f"Running in Calibration Mode (Null Mode: {args.cfg_null_mode})")
    
    print(f"Starting Trajectory Stability Calibration on {args.calib_batches} batches...")
    
    # 用于累积统计量的变量 (均值和方差的在线计算)
    # 目标：计算 sigma_c = Std(Delta_c)
    sum_delta = None    # [C]
    sum_sq_delta = None # [C]
    n_samples = 0       # Total pixels count (Batch * TimeSteps * H * W)
    
    # 主循环
    for i, batch in enumerate(tqdm(dataloader, total=args.calib_batches)):
        if i >= args.calib_batches:
            break
            
        imgs_hr = batch[0].to(device)
        if args.lr_data_path:
            imgs_lr = batch[1].to(device)
        else:
            imgs_lr = degradation_model(imgs_hr, scale=None)
        
        if torch.isnan(imgs_hr).any() or torch.isnan(imgs_lr).any():
            continue

        # VAE Encoding
        with torch.no_grad():
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

        # 构建 Condition Tokens (有条件)
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

        # MAE Forward 生成 z_cond (有条件)
        x = model.forward_mae_encoder(
            hr_tokens, mask, lr_tokens, shape_hr, shape_lr,
            cond_tokens=cond_tokens_encoder
        )
        
        # [修改点 1] 添加 _ 接收多余的返回值
        z, pos_embed, _ = model.forward_mae_decoder(
            x, mask, shape_hr, shape_lr,
            cond_tokens=cond_tokens_decoder
        )
        
        # 提取被遮挡部分
        mask_bool = mask.bool()
        indices_to_pred = mask_bool.nonzero(as_tuple=True)
        z_sub = z[indices_to_pred]
        pos_sub = pos_embed[indices_to_pred]
        z_cond = z_sub + pos_sub
        if swinir_model is not None:
            with torch.cuda.amp.autocast():
                imgs_lr = preprocess_with_swinir(imgs_lr, swinir_model)
        if torch.isnan(z_cond).any():
             continue

        # === CFG 无条件分支构造 ===
        z_uncond = None
        if args.cfg_scale > 1.0:
            # 1. 构造全 0 输入 (无条件)
            x_lr_zeros = torch.zeros_like(x_lr)
            lr_tokens_zeros = torch.zeros_like(lr_tokens)
            
            # 2. 生成无条件 Condition Tokens
            cond_enc_un, cond_dec_un = model._build_lr_cond_tokens(x_lr_zeros, lr_tokens=lr_tokens_zeros)
            
            # 3. 运行无条件分支的 MAE (Encoder + Decoder)
            # 关键：mask 必须和有条件分支保持完全一致
            x_un = model.forward_mae_encoder(
                hr_tokens, mask, lr_tokens_zeros, shape_hr, shape_lr,
                cond_tokens=cond_enc_un
            )
            
            # [修改点 2] 统一使用 3 变量解包
            z_un_full, pos_un_full, _ = model.forward_mae_decoder(
                x_un, mask, shape_hr, shape_lr,
                cond_tokens=cond_dec_un
            )
            
            # 4. 提取对应位置的 Latent
            z_sub_un = z_un_full[indices_to_pred]
            pos_sub_un = pos_un_full[indices_to_pred]
            z_uncond = z_sub_un + pos_sub_un
        # ================================

        # === 运行采样并收集 Delta ===
        collector = DeltaCollector(window_steps)
        
        # 传入 collector 和 CFG 参数到采样函数
        model.diffloss.sample(
            z_cond, 
            temperature=args.temperature, 
            confidence_accumulator=collector, 
            return_confidence=False,
            cfg_scale=args.cfg_scale,   
            z_uncond=z_uncond           
        )
        
        if not collector.deltas:
            continue
            
        for delta_batch in collector.deltas:
            # delta_batch: [N, D]
            D = delta_batch.shape[1]
            
            # 转置为 [Channels, N_masked_tokens]
            flat_delta = delta_batch.transpose(0, 1).double() 
            curr_n = flat_delta.shape[1]
            
            if sum_delta is None:
                sum_delta = torch.zeros(D, dtype=torch.float64)
                sum_sq_delta = torch.zeros(D, dtype=torch.float64)
            
            # 累积 Sum 和 SumSq
            sum_delta += flat_delta.sum(dim=1)
            sum_sq_delta += (flat_delta ** 2).sum(dim=1)
            n_samples += curr_n

    # 6. 计算最终统计量
    if n_samples > 0:
        # Mean = Sum / N
        mean = sum_delta / n_samples
        # Var = E[x^2] - (E[x])^2
        var = (sum_sq_delta / n_samples) - (mean ** 2)
        
        # 数值稳定性处理
        var = torch.relu(var)
        std = torch.sqrt(var).float()
        
        sigma_min = 1e-4
        std = torch.clamp(std, min=sigma_min)
        
        sigma_delta_np = std.numpy()
        
        print("-" * 30)
        print(f"Calibration Completed (N={n_samples} pixels per channel)")
        print(f"Sigma_delta (per channel): \n{sigma_delta_np}")
        print("-" * 30)
        
        # 保存
        os.makedirs(args.output_dir, exist_ok=True)
        conf_window_str = args.conf_window.replace(":", "_") if args.conf_window else "all"
        
        # 文件名带上 cfg_scale 和 temperature，方便管理
        save_name = f"confidence_stats_cfg{args.cfg_scale:.1f}_temp{args.temperature:.2f}.npz"
        save_path = os.path.join(args.output_dir, save_name)
        
        np.savez(save_path, sigma_delta=sigma_delta_np)
        print(f"Statistics saved to: {save_path}")
        print(f"Usage: Please use --conf_window '{args.conf_window}' during inference.")
        
    else:
        print("Error: No valid data collected. Please check if 'conf_window' intersects with diffusion steps.")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)