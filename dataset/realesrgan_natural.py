import torch
import numpy as np
import cv2
import random
import math
from scipy import special

from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
    random_add_poisson_noise,
)

def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """生成 2D Sinc 滤波器 (用于 Ringing artifacts 模拟)"""
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel

class RealESRGANDegradationNatural:
    """
    完全对齐 VARSR (params_realesrgan.yml) 的退化逻辑
    """
    def __init__(self, device='cuda'):
        self.device = device
        
        # --- 第一阶段参数 (First Stage) ---
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]
        self.sinc_prob = 0.1
        
        self.resize_prob = [0.2, 0.7, 0.1] # up, down, keep
        self.resize_range = [0.3, 1.5]     
        
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 15]         
        self.poisson_scale_range = [0.05, 2.0] 
        self.gray_noise_prob = 0.4
        
        self.jpeg_range = [60, 95]         

        # --- 第二阶段参数 (Second Stage) ---
        self.second_blur_prob = 0.5        
        
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1

        self.resize_prob2 = [0.3, 0.4, 0.3]
        self.resize_range2 = [0.6, 1.2]    
        
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 12]        
        self.poisson_scale_range2 = [0.05, 1.0]
        self.gray_noise_prob2 = 0.4
        
        self.jpeg_range2 = [60, 100]       

        self.final_sinc_prob = 0.8
        
        self.kernel_range = [7, 9, 11, 13, 15, 17, 19, 21]
        self.pulse_tensor = np.zeros((21, 21), dtype=np.float32)
        self.pulse_tensor[10, 10] = 1

    @torch.no_grad()
    def __call__(self, imgs_hr, scale=4):
        """
        imgs_hr: Tensor (B, C, H, W) range [-1, 1] or [0, 1]
        """
        if isinstance(imgs_hr, list):
             pass

        device = imgs_hr.device
        if scale is None: scale = 4
        
        # Tensor [-1, 1] -> Numpy [0, 1] (B, H, W, C)
        if imgs_hr.min() < 0:
            imgs_np = (imgs_hr.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2
        else:
            imgs_np = imgs_hr.cpu().numpy().transpose(0, 2, 3, 1)
            
        imgs_np = np.clip(imgs_np, 0, 1).astype(np.float32)
        b, h, w, c = imgs_np.shape
        
        imgs_lr_list = []

        resize_modes = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA] 

        for i in range(b):
            img = imgs_np[i].copy()
            
            # ================== 第一阶段退化 (First Stage) ==================
            # 1.1 Blur
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list, self.kernel_prob, kernel_size,
                    self.blur_sigma, self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range, self.betap_range, noise_range=None
                )
            img = cv2.filter2D(img, -1, kernel)

            # 1.2 Random Resize
            updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
            if updown_type == 'up':
                scale_v = np.random.uniform(1, self.resize_range[1])
            elif updown_type == 'down':
                scale_v = np.random.uniform(self.resize_range[0], 1)
            else:
                scale_v = 1
            mode = random.choice(resize_modes)
            if scale_v != 1:
                img = cv2.resize(img, (int(w * scale_v), int(h * scale_v)), interpolation=mode)

            # 1.3 Add Noise (Gaussian or Poisson)
            if np.random.uniform() < self.gaussian_noise_prob:
                img = random_add_gaussian_noise(
                    img, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob)
            else:
                img = random_add_poisson_noise(
                    img, scale_range=self.poisson_scale_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob)

            # 1.4 JPEG Compression
            if np.random.uniform() < 0.9: 
                img = random_add_jpg_compression(img, quality_range=self.jpeg_range)

            # ================== 第二阶段退化 (Second Stage) ==================
            # 2.1 Blur (Conditional)
            if np.random.uniform() < self.second_blur_prob:
                kernel_size = random.choice(self.kernel_range)
                if np.random.uniform() < self.sinc_prob2:
                    if kernel_size < 13:
                        omega_c = np.random.uniform(np.pi / 3, np.pi)
                    else:
                        omega_c = np.random.uniform(np.pi / 5, np.pi)
                    kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
                else:
                    kernel2 = random_mixed_kernels(
                        self.kernel_list2, self.kernel_prob2, kernel_size,
                        self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi],
                        self.betag_range2, self.betap_range2, noise_range=None
                    )
                img = cv2.filter2D(img, -1, kernel2)

            # 2.2 Random Resize (Intermediate)
            updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
            if updown_type == 'up':
                scale_v = np.random.uniform(1, self.resize_range2[1])
            elif updown_type == 'down':
                scale_v = np.random.uniform(self.resize_range2[0], 1)
            else:
                scale_v = 1
            
            mode = random.choice(resize_modes)
            current_h, current_w = img.shape[:2]
            if scale_v != 1:
                img = cv2.resize(img, (int(current_w * scale_v), int(current_h * scale_v)), interpolation=mode)

            # 2.3 Add Noise (Gaussian or Poisson)
            if np.random.uniform() < self.gaussian_noise_prob2:
                img = random_add_gaussian_noise(
                    img, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob2)
            else:
                img = random_add_poisson_noise(
                    img, scale_range=self.poisson_scale_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob2)

            # ================== 最终处理 (Final Block: Resize+Sinc & JPEG) ==================
            target_h, target_w = int(h // scale), int(w // scale)
            
            # 准备 Sinc Kernel
            if np.random.uniform() < self.final_sinc_prob:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            else:
                sinc_kernel = self.pulse_tensor

            # 随机交换顺序
            if np.random.uniform() < 0.5:
                # 顺序 A: [Resize back + Sinc] -> JPEG
                mode = random.choice(resize_modes)
                img = cv2.resize(img, (target_w, target_h), interpolation=mode)
                img = cv2.filter2D(img, -1, sinc_kernel)
                
                if np.random.uniform() < 0.9:
                    img = random_add_jpg_compression(img, quality_range=self.jpeg_range2)
            else:
                # 顺序 B: JPEG -> [Resize back + Sinc]
                if np.random.uniform() < 0.9:
                    img = random_add_jpg_compression(img, quality_range=self.jpeg_range2)
                
                mode = random.choice(resize_modes)
                img = cv2.resize(img, (target_w, target_h), interpolation=mode)
                img = cv2.filter2D(img, -1, sinc_kernel)

            # ================== 输出处理 ==================
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            imgs_lr_list.append(img)

        # Stack -> Tensor -> [-1, 1]
        imgs_lr = np.stack(imgs_lr_list, axis=0)
        imgs_lr = imgs_lr.transpose(0, 3, 1, 2) # (B, C, H, W)
        imgs_lr = torch.from_numpy(imgs_lr.copy()).to(device)
        
        imgs_lr = imgs_lr * 2 - 1
        imgs_lr = torch.clamp(imgs_lr, -1, 1)

        return imgs_lr