import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import numpy as np
import cv2
import math

# 引入degradation.py 中的核心函数
from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)

class SRDataset(Dataset):
    def __init__(self, root, hr_size=128, lr_size=32, is_train=True):
        self.dataset = ImageFolder(root)
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.is_train = is_train
        
        # ============================================================
        # CodeFormer 的默认退化配置 (参考自 codeformer.py)
        # ============================================================
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3]
        self.downsample_range = [1, 12] # 随机下采样倍率范围
        self.noise_range = [0, 15]
        self.jpeg_range = [30, 95]

        # 归一化工具
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __getitem__(self, index):
        # 1. 读取 HR 图片 (PIL Image)
        img_hr, _ = self.dataset[index]

        # 2. 裁剪与增强 (HR) - 保持原有逻辑
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                img_hr, output_size=(self.hr_size, self.hr_size)
            )
            img_hr = TF.crop(img_hr, i, j, h, w)
            if random.random() > 0.5:
                img_hr = TF.hflip(img_hr)
        else:
            img_hr = TF.center_crop(img_hr, (self.hr_size, self.hr_size))

        # 转换: PIL -> Numpy (H, W, C) [0, 1] RGB
        # degradation.py 需要 float32 且范围 [0, 1] 的输入
        img_hr_np = np.array(img_hr).astype(np.float32) / 255.0
        
        # 3. 生成 LR (CodeFormer 风格退化)
        if self.is_train:
            h, w, _ = img_hr_np.shape
            
            # --- A. 随机模糊 (Blur) ---
            # 生成混合模糊核
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None
            )
            # 应用模糊
            img_lq = cv2.filter2D(img_hr_np, -1, kernel)
            
            # --- B. 随机下采样 (Resize) ---
            # 模拟不规则的低分辨率
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
            
            # --- C. 随机噪声 (Noise) ---
            # 50% 概率添加高斯噪声
            # noise_range [0, 15] 对应 0-255 范围，工具函数内部会处理
            if self.noise_range is not None: # CodeFormer 逻辑：始终添加或按概率添加，这里参考原版
                img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
                
            # --- D. JPEG 压缩 ---
            if self.jpeg_range is not None:
                img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
            
            # --- E. 强制调整到目标 LR 尺寸 (32x32) ---
            # 无论中间怎么折腾，最后必须缩放到模型需要的尺寸
            img_lq = cv2.resize(img_lq, (self.lr_size, self.lr_size), interpolation=cv2.INTER_LINEAR)

        else:
            # 验证集：保持纯净，只做标准双三次下采样
            # 这样可以评估模型在标准 degradation 下的性能
            img_lq = cv2.resize(img_hr_np, (self.lr_size, self.lr_size), interpolation=cv2.INTER_CUBIC)

        # 4. 转 Tensor 并归一化 ([-1, 1])
        # Numpy (H,W,C) -> Tensor (C,H,W)
        # 此时 img_hr_np 和 img_lq 都是 [0, 1] 范围的 numpy
        hr_tensor = torch.from_numpy(np.ascontiguousarray(img_hr_np.transpose(2, 0, 1))).float()
        lr_tensor = torch.from_numpy(np.ascontiguousarray(img_lq.transpose(2, 0, 1))).float()

        hr_tensor = self.normalize(hr_tensor)
        lr_tensor = self.normalize(lr_tensor)

        return hr_tensor, lr_tensor

    def __len__(self):
        return len(self.dataset)