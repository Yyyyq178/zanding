import os
import cv2
import math
import numpy as np
import random
import argparse
from pathlib import Path

# 引入底层的退化函数 (确保 dataset/degradation.py 存在)
from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
    random_add_poisson_noise,
    circular_lowpass_kernel
)

class DegradationSimulator:
    def __init__(self):
        # ============================================================
        # Real-ESRGAN 默认参数配置 (与 dataset_sr.py 保持一致)
        # ============================================================
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob = 0.1
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]

        self.resize_prob = [0.2, 0.7, 0.1]
        self.resize_range = [0.15, 1.5]
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4

        # Stage 2
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.resize_prob2 = [0.2, 0.7, 0.1]
        self.resize_range2 = [0.3, 1.2]
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4

        self.final_sinc_prob = 0.8
        self.jpeg_range2 = [30, 95]
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

    def apply_degradation(self, img_hr, scale_factor=4.0):
        """
        对 HR 图片应用 Real-ESRGAN 退化，并缩放到指定倍率
        img_hr: Numpy array (H, W, C), BGR format, [0, 1] float
        scale_factor: 下采样倍率 (例如 4.0 表示缩小 4 倍)
        """
        h_hr, w_hr, _ = img_hr.shape
        out = img_hr.copy()

        # 计算目标 LR 尺寸
        h_lr = int(h_hr / scale_factor)
        w_lr = int(w_hr / scale_factor)

        # ------------------------ Stage 1 ------------------------
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

        # Random Resize
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

        # ------------------------ Stage 2 ------------------------
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

        # Random Resize
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
            out = cv2.resize(out, (int(w_cur * scale), int(h_cur * scale)), interpolation=mode)

        # Noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise(
                out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob2
            )
        else:
            out = random_add_poisson_noise(
                out, scale_range=self.poisson_scale_range2, gray_prob=self.gray_noise_prob2, clip=True, rounds=False
            )

        # ------------------------ Final ------------------------
        # Sinc & Final Resize to Target Size
        mode = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            out = cv2.filter2D(out, -1, sinc_kernel)
            
        # 强制调整到目标 LR 尺寸
        out = cv2.resize(out, (w_lr, h_lr), interpolation=mode)

        # JPEG
        if np.random.uniform() < 0.5:
            out = random_add_jpg_compression(out, self.jpeg_range2)
            
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input HR image or folder')
    parser.add_argument('--output', type=str, default='test_lr_images', help='Output folder')
    parser.add_argument('--scale', type=float, default=4.0, help='Downsampling scale (e.g. 4.0 for 4x downsample)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    simulator = DegradationSimulator()

    # 处理单张图片或文件夹
    if os.path.isfile(args.input):
        img_paths = [args.input]
    else:
        img_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"🚀 Generating LR images with Real-ESRGAN degradation (Scale: {args.scale}x)...")
    
    for path in img_paths:
        img_name = os.path.basename(path)
        # 读取 HR
        img = cv2.imread(path).astype(np.float32) / 255.0
        
        # 生成 LR
        img_lr = simulator.apply_degradation(img, scale_factor=args.scale)
        
        # 保存
        save_path = os.path.join(args.output, f"{os.path.splitext(img_name)[0]}_lr.png")
        # 转回 0-255 uint8
        img_lr = (np.clip(img_lr, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(save_path, img_lr)
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    main()