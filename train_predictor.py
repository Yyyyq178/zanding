import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision.utils as vutils

# === 加载你项目中的模块 ===
from models.vae import AutoencoderKL
from dataset.dataset_paired import PairedSRDataset

# === 1. 定义难度预测器 (Difficulty Predictor) ===
class DifficultyPredictor(nn.Module):
    def __init__(self, in_channels=16, hidden_dim=64):
        """
        in_channels: VAE Latent的维度 (你的配置里是 16)
        输出: 1通道的 Heatmap (表示该位置的修复难度/预期误差)
        """
        super().__init__()
        self.net = nn.Sequential(
            # Input: LR Latent Only [B, 16, H, W]
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            
            # Output: Error Map (1 channel)
            nn.Conv2d(hidden_dim, 1, 1), 
            # 关键修改：用 ReLU 保证输出非负 (误差 >= 0)，而不是 Sigmoid
            nn.ReLU() 
        )

    def forward(self, lr_latent):
        return self.net(lr_latent)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_path', type=str, required=True, help='Path to HR dataset')
    parser.add_argument('--lr_path', type=str, required=True, help='Path to LR dataset')
    parser.add_argument('--vae_path', type=str, default='pretrained_models/vae/kl16.ckpt')
    parser.add_argument('--output_dir', type=str, default='output_predictor_fixed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=256) # 建议与训练 MAR 时一致
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    # 根据你的 engine_mar.py，Latent 缩放因子是 0.2325
    parser.add_argument('--vae_scale', type=float, default=0.2325) 
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')

    # === 2. 加载 VAE (冻结) ===
    print(f"Loading VAE from {args.vae_path}...")
    # 根据你的 train_sr.sh，你的 embed_dim 是 16
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).to(device).eval()
    for p in vae.parameters(): p.requires_grad = False

    # === 3. 初始化预测器 ===
    # 输入维度是 16 (VAE latent dim)
    model = DifficultyPredictor(in_channels=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 使用 L1 Loss 来回归误差 (L1 比 MSE 对边缘更加敏感，适合预测纹理难度)
    criterion = nn.L1Loss()

    # === 4. 数据集 ===
    print(f"Loading Paired Dataset...\n HR: {args.hr_path}\n LR: {args.lr_path}")
    dataset = PairedSRDataset(root_hr=args.hr_path, root_lr=args.lr_path, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    print("Start Training Predictor...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, (x_hr, x_lr, _) in enumerate(dataloader):
            x_hr = x_hr.to(device)
            x_lr = x_lr.to(device)
            
            with torch.no_grad():
                # 获取 Latent (包含缩放因子)
                lat_hr = vae.encode(x_hr).sample() * 0.2325
                lat_lr = vae.encode(x_lr).sample() * 0.2325
                
                # 1. 计算 GT (真实难度): Channel 平均后的绝对误差
                gt_difficulty = (lat_hr - lat_lr).abs().mean(dim=1, keepdim=True)

            # 2. 预测难度
            pred_difficulty = model(lat_lr)
            
            # 3. 计算 Loss (L1 Loss)
            loss = criterion(pred_difficulty, gt_difficulty)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} [{i}/{len(dataloader)}] Loss: {loss.item():.6f}")

        # === 1. 修复图片全白问题 (Visualization) ===
        # 每个 epoch 结束保存一次预览图
        with torch.no_grad():
            # (A) LR图: 反归一化 [-1, 1] -> [0, 1]
            vis_lr = (x_lr[:4] + 1) / 2.0
            
            # (B) GT图: 插值放大
            vis_gt = F.interpolate(gt_difficulty[:4], size=(args.img_size, args.img_size), mode='nearest')
            vis_gt = vis_gt.repeat(1, 3, 1, 1) # 转3通道
            
            # (C) 预测图: 插值放大
            vis_pred = F.interpolate(pred_difficulty[:4], size=(args.img_size, args.img_size), mode='nearest')
            vis_pred = vis_pred.repeat(1, 3, 1, 1) # 转3通道
            
            # 拼接
            debug_batch = torch.cat([vis_lr, vis_gt, vis_pred], dim=0)
            
            # 保存预览图
            # [关键] normalize=True: 自动将最小值映射为0(黑)，最大值映射为1(白)，解决过曝/全白问题
            # [关键] scale_each=True: 每张小图独立归一化，防止某张图极亮导致其他图全黑
            vutils.save_image(
                debug_batch, 
                f"{args.output_dir}/debug_preview.png", # 也可以固定名字，每次覆写
                nrow=4, 
                normalize=True, 
                scale_each=True
            )

        # === 2. 修复权重堆积问题 (Save Model) ===
        # 使用固定文件名 'predictor_latest.pth'，每次自动覆写上一次的文件
        save_path = f"{args.output_dir}/predictor_latest.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch} finished. Model overwritten to: {save_path}")

    print("Training Done.")

if __name__ == '__main__':
    main()