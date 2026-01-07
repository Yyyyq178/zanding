import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vae import AutoencoderKL
# [修正] 导入您提供的 PairedSRDataset
from dataset.dataset_paired import PairedSRDataset
import torch.nn.functional as F

# === 1. 定义小网络 (裁判) ===
class LatentConfidencePredictor(nn.Module):
    def __init__(self, in_channels=16, hidden_dim=64):
        super().__init__()
        # 输入: SR(16) + LR(16) = 32通道
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid() 
        )

    def forward(self, sr, lr):
        x = torch.cat([sr, lr], dim=1)
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_path', type=str, required=True, help='Path to HR dataset')
    parser.add_argument('--lr_path', type=str, required=True, help='Path to LR dataset (Paired)')
    parser.add_argument('--vae_path', type=str, default='pretrained_models/vae/kl16.ckpt')
    parser.add_argument('--output_dir', type=str, default='output_predictor')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=256, help='Image size for training (crop/resize)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')

    # === 2. 加载 VAE (冻结) ===
    print("Loading VAE...")
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).to(device).eval()
    for p in vae.parameters(): p.requires_grad = False

    # === 3. 初始化预测器 ===
    model = LatentConfidencePredictor(in_channels=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # === 4. 数据集 (使用 PairedSRDataset) ===
    print(f"Loading Paired Dataset from:\n HR: {args.hr_path}\n LR: {args.lr_path}")
    # PairedSRDataset 会自动处理 HR 和 LR 的同步裁剪/缩放
    # 注意：确保您的 PairedSRDataset 返回的 LR 已经被 Resize 到和 HR 一样大 (您的代码逻辑里是这样写的)
    dataset = PairedSRDataset(root_hr=args.hr_path, root_lr=args.lr_path, img_size=args.img_size)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    print("Start Training Predictor...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # DataLoader 返回: (img_hr, img_lr, filename)
        for i, (x_hr, x_lr, _) in enumerate(dataloader):
            x_hr = x_hr.to(device)
            x_lr = x_lr.to(device)
            
            with torch.no_grad():
                # 直接编码真实的 HR 和 LR
                # 注意：PairedSRDataset 已经把 LR resize 到了 img_size，所以可以直接进 VAE
                lat_hr = vae.encode(x_hr).sample() * 0.18215
                lat_lr = vae.encode(x_lr).sample() * 0.18215

            # === 构造正负样本 ===
            
            # 1. 正样本: (Real HR, Real LR) -> 标签 1.0
            # 这次我们用的是真实的成对数据，模型会学习到真实 LR 应对应的 HR 细节
            pred_pos = model(lat_hr, lat_lr)
            loss_pos = criterion(pred_pos, torch.ones_like(pred_pos))

            # 2. 负样本: (Noisy HR, Real LR) -> 标签 0.0
            # 模拟：Latent 是对的，但是含有噪声/伪影
            noise = torch.randn_like(lat_hr) * 0.5 
            lat_fake = lat_hr + noise
            pred_neg = model(lat_fake, lat_lr)
            loss_neg = criterion(pred_neg, torch.zeros_like(pred_neg))
            
            # 3. (可选) 负样本: (Mismatch HR, Real LR) -> 标签 0.0
            # 模拟：纹理清晰但语义不对 (猫头对狗嘴)
            lat_shuffled = torch.roll(lat_hr, shifts=1, dims=0)
            pred_shuffle = model(lat_shuffled, lat_lr)
            loss_shuffle = criterion(pred_shuffle, torch.zeros_like(pred_shuffle))
            
            loss = loss_pos + loss_neg + loss_shuffle
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")

        # 保存
        torch.save(model.state_dict(), f"{args.output_dir}/predictor_epoch{epoch}.pth")
        print(f"Saved: {args.output_dir}/predictor_epoch{epoch}.pth")

if __name__ == '__main__':
    main()