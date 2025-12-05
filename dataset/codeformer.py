import torch
import torchvision.transforms.functional as TF
# 复用你现有的噪声函数
from dataset.degradation import random_add_gaussian_noise_pt

def degradation_codeformer(imgs_hr, target_size):
    """
    在 GPU 上实现 CodeFormer 的动态退化：
    HR -> (随机模糊) -> 直接缩放到 Target Size -> (随机噪声) -> LR
    
    Args:
        imgs_hr: [B, C, H, W], 归一化范围 [-1, 1]
        target_size: int, 目标 LR 的边长
    """
    # 1. 反归一化：从 [-1, 1] 转换到 [0, 1]，因为后续噪声函数通常基于 0-1
    imgs = (imgs_hr + 1.0) * 0.5 
    
    # === A. Random Blur (50% 概率) ===
    # 模拟 CodeFormer 配置: sigma [0.1, 12], kernel_size 41
    if torch.rand(1).item() < 0.5:
        sigma = torch.empty(1).uniform_(0.1, 12.0).item()
        # 使用 PyTorch 原生的高斯模糊，支持 Batch 操作
        imgs = TF.gaussian_blur(imgs, kernel_size=41, sigma=[sigma, sigma])

    # === B. Resize (一步到位下采样) ===
    # 直接从当前尺寸 (例如 256) 缩放到 动态目标尺寸 (例如 83)
    # 开启 antialias 防止锯齿
    imgs = torch.nn.functional.interpolate(
        imgs, 
        size=(target_size, target_size), 
        mode='bilinear', 
        align_corners=False,
        antialias=True
    )

    # === C. Random Gaussian Noise (50% 概率) ===
    # 模拟 CodeFormer 配置: noise_range [0, 15] (基于 255 范围)
    if torch.rand(1).item() < 0.5:
        imgs = random_add_gaussian_noise_pt(
            imgs, 
            sigma_range=(0, 15/255.0), # 转换为 0-1 范围
            clip=True, 
            rounds=False, 
            gray_prob=0
        )

    # 2. 归一化：恢复到 [-1, 1] 给模型使用
    imgs = (imgs - 0.5) / 0.5
    
    return imgs