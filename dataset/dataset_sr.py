import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import numpy as np
import cv2
import math
import os

from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
    random_add_poisson_noise,
    circular_lowpass_kernel
)

class SRDataset(Dataset):
    def __init__(self, root, hr_size=256, lr_size=64, is_train=True, degradation_type='codeformer'):
        """
        Args:
            degradation_type (str): 'codeformer' (一阶退化) 或 'realesrgan' (高阶退化)
        """
        self.dataset = ImageFolder(root)
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.is_train = is_train
        self.degradation_type = degradation_type.lower()
        
        print(f"Dataset initialized with degradation: {self.degradation_type}")

        # 归一化工具
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # ============================================================
        # 1. CodeFormer 参数配置 (简单退化)
        # ============================================================
        if self.degradation_type == 'codeformer':
            self.cf_blur_kernel_size = 41
            self.cf_kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
            self.cf_kernel_prob = [0.5, 0.5, 0.12, 0.03, 0.12, 0.03]
            self.cf_blur_sigma = [0.1, 12]
            self.cf_downsample_range = [1, 12]
            self.cf_noise_range = [0, 15]
            self.cf_jpeg_range = [30, 100]

        # ============================================================
        # 2. Real-ESRGAN 参数配置 (高阶退化)
        # ============================================================
        elif self.degradation_type == 'realesrgan':
            # 第一阶段
            self.blur_kernel_size = 21
            self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
            self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
            self.sinc_prob = 0.1
            self.blur_sigma = [0.2, 3]
            self.betag_range = [0.5, 4]
            self.betap_range = [1, 2]
            self.resize_prob = [0.2, 0.7, 0.1]
            self.resize_range = [0.3, 1.5]
            self.gaussian_noise_prob = 0.5
            self.noise_range = [1, 15]
            self.poisson_scale_range = [0.05, 2]
            self.gray_noise_prob = 0.4
            self.jpeg_range2 = [60, 95]
            
            # 第二阶段
            self.blur_kernel_size2 = 21
            self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
            self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
            self.sinc_prob2 = 0.1
            self.blur_sigma2 = [0.2, 3]
            self.betag_range2 = [0.5, 4]
            self.betap_range2 = [1, 2]
            self.resize_prob2 = [0.3, 0.4, 0.3]
            self.resize_range2 = [0.6, 1.2]
            self.gaussian_noise_prob2 = 0.5
            self.noise_range2 = [1, 12]
            self.poisson_scale_range2 = [0.05, 1.0]
            self.gray_noise_prob2 = 0.4
            self.jpeg_range2 = [60, 100]
            
            # 最终处理
            self.final_sinc_prob = 0.8
            self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        
        else:
            raise ValueError(f"Unsupported degradation type: {self.degradation_type}")

    def __getitem__(self, index):
        # 1. 读取 HR 图片
        img_hr, _ = self.dataset[index]

        # 2. 裁剪与增强 (HR)
        if self.is_train:
            if img_hr.size[0] > self.hr_size and img_hr.size[1] > self.hr_size:
                i, j, h, w = transforms.RandomCrop.get_params(
                    img_hr, output_size=(self.hr_size, self.hr_size)
                )
                img_hr = TF.crop(img_hr, i, j, h, w)
            else:
                img_hr = TF.resize(img_hr, (self.hr_size, self.hr_size))
            
            if random.random() > 0.5:
                img_hr = TF.hflip(img_hr)
        else:
            img_hr = TF.center_crop(img_hr, (self.hr_size, self.hr_size))

        # 转换: PIL -> Numpy (H, W, C) [0, 1] RGB, float32
        img_hr_np = np.array(img_hr).astype(np.float32) / 255.0
        
        # 3. 生成 LR
        img_lq = img_hr_np.copy()
        
        # if self.degradation_type == 'codeformer':
        #     img_lq = self._getitem_codeformer(img_hr_np)
        # elif self.degradation_type == 'realesrgan':
        #     img_lq = self._getitem_realesrgan(img_hr_np)
        

        # 4. 转 Tensor 并归一化 ([-1, 1])
        hr_tensor = torch.from_numpy(np.ascontiguousarray(img_hr_np.transpose(2, 0, 1))).float()
        lr_tensor = torch.from_numpy(np.ascontiguousarray(img_lq.transpose(2, 0, 1))).float()

        hr_tensor = self.normalize(hr_tensor)
        lr_tensor = self.normalize(lr_tensor)

        path, _ = self.dataset.samples[index]
        filename = os.path.basename(path)
        
        return hr_tensor, lr_tensor, filename 

    def _getitem_codeformer(self, img_hr):
        """CodeFormer 的一阶退化逻辑"""
        h, w, _ = img_hr.shape
        
        # Blur
        kernel = random_mixed_kernels(
            self.cf_kernel_list, self.cf_kernel_prob, self.cf_blur_kernel_size,
            self.cf_blur_sigma, self.cf_blur_sigma, [-math.pi, math.pi], noise_range=None
        )
        out = cv2.filter2D(img_hr, -1, kernel)
        
        # Resize
        scale = np.random.uniform(self.cf_downsample_range[0], self.cf_downsample_range[1])
        out = cv2.resize(out, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        
        # Noise
        if self.cf_noise_range is not None:
            out = random_add_gaussian_noise(out, self.cf_noise_range)
            
        # JPEG
        if self.cf_jpeg_range is not None:
            out = random_add_jpg_compression(out, self.cf_jpeg_range)
        
        # Final Resize
        out = cv2.resize(out, (self.lr_size, self.lr_size), interpolation=cv2.INTER_LINEAR)
        return out

    def _getitem_realesrgan(self, img_hr):
        """Real-ESRGAN 的高阶退化逻辑"""
        h_hr, w_hr, _ = img_hr.shape
        out = img_hr.copy()
        
        # --- Stage 1 ---
        # Blur
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
        out = cv2.filter2D(out, -1, kernel)

        # Resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
        if scale != 1:
            out = cv2.resize(out, (int(w_hr * scale), int(h_hr * scale)), interpolation=mode)

        # Noise
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise(
                out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob
            )
        else:
            out = random_add_poisson_noise(
                out, scale_range=self.poisson_scale_range, gray_prob=self.gray_noise_prob, clip=True, rounds=False
            )

        # --- Stage 2 ---
        # Blur
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
        out = cv2.filter2D(out, -1, kernel2)

        # Resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
        if scale != 1:
            h_cur, w_cur, _ = out.shape
            new_h, new_w = int(h_cur * scale), int(w_cur * scale)
            # 保护措施：不小于 LR 尺寸的一半
            new_h = max(new_h, self.lr_size // 2)
            new_w = max(new_w, self.lr_size // 2)
            out = cv2.resize(out, (new_w, new_h), interpolation=mode)

        # Noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise(
                out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob2
            )
        else:
            out = random_add_poisson_noise(
                out, scale_range=self.poisson_scale_range2, gray_prob=self.gray_noise_prob2, clip=True, rounds=False
            )

        # --- Final ---
        # Sinc & Resize
        mode = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            out = cv2.filter2D(out, -1, sinc_kernel)
            
        out = cv2.resize(out, (self.lr_size, self.lr_size), interpolation=mode)

        # JPEG
        if np.random.uniform() < 0.5:
            out = random_add_jpg_compression(out, self.jpeg_range2)
            
        return out

    def __len__(self):
        return len(self.dataset)