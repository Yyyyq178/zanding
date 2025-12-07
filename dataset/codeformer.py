import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2

from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise_pt,
    random_add_jpg_compression
)

class CodeFormerDegradation:
    """
    HR[-1,1] → Blur → Down → Noise → JPEG → Up → LR[-1,1]
    """

    def __init__(
        self,
        blur_kernel_size=41,
        kernel_list=("iso", "aniso", "generalized_iso", "generalized_aniso"),
        kernel_prob=(0.5, 0.5, 0.25, 0.25),
        blur_sigma=(0.1, 12),
        downsample_range=(2, 4),
        noise_range=(0, 15/255.0),
        jpeg_range=(30, 100),
        blur_prob=1.0,
        noise_prob=1.0,
        jpeg_prob=1.0,
        noise_gray_prob=0.0
    ):
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.jpeg_prob = jpeg_prob
        self.noise_gray_prob = noise_gray_prob

    @torch.no_grad()
    def __call__(self, imgs_hr, scale=None):
        """
        imgs_hr: [B,C,H,W], range [-1,1]
        """
        b, c, h, w = imgs_hr.shape

        # Convert to [0,1]
        imgs = (imgs_hr + 1) * 0.5

        # Random scale
        if scale is None:
            # 使用初始化时定义的范围
            scale = random.uniform(*self.downsample_range)
        else:
            scale = scale

        h_lr = int(h // scale)
        w_lr = int(w // scale)

        # ----------------------------------------------------
        # Step 1. Blur (Mixed Kernels)
        # ----------------------------------------------------
        if random.random() < self.blur_prob:

            # numpy kernel → torch depthwise conv kernel
            kernel_np = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size=self.blur_kernel_size,
                sigma_x_range=self.blur_sigma,
                sigma_y_range=self.blur_sigma,
                rotation_range=(-np.pi, np.pi),
                noise_range=None
            )

            kernel = torch.tensor(kernel_np, device=imgs.device, dtype=imgs.dtype)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

            # Depthwise blur for all channels
            imgs = torch.nn.functional.conv2d(
                imgs,
                kernel.repeat(c, 1, 1, 1),
                padding=self.blur_kernel_size // 2,
                groups=c
            )

        # ----------------------------------------------------
        # Step 2. Downsample
        # ----------------------------------------------------
        imgs_lr = F.interpolate(
            imgs,
            size=(h_lr, w_lr),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )

        # ----------------------------------------------------
        # Step 3. Gaussian Noise
        # ----------------------------------------------------
        if random.random() < self.noise_prob:
            imgs_lr = random_add_gaussian_noise_pt(
                imgs_lr,
                sigma_range=self.noise_range,
                gray_prob=self.noise_gray_prob,
                clip=True,
                rounds=False
            )

        # ----------------------------------------------------
        # Step 4. JPEG compression 
        # Tensor → numpy → JPEG → Tensor
        # ----------------------------------------------------
        if random.random() < self.jpeg_prob:

            imgs_list = []
            for i in range(b):
                img_np = imgs_lr[i].permute(1, 2, 0).cpu().numpy()  # HWC

                img_np = random_add_jpg_compression(
                    img_np,
                    quality_range=self.jpeg_range
                )

                img_t = torch.tensor(img_np, dtype=imgs_lr.dtype).permute(2, 0, 1)
                imgs_list.append(img_t)

            imgs_lr = torch.stack(imgs_list, dim=0).to(imgs_lr.device)

        # ----------------------------------------------------
        # Step 5. Upsample back to HR size
        # ----------------------------------------------------
        imgs_final = F.interpolate(
            imgs_lr,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )

        # ----------------------------------------------------
        # Step 6. Back to [-1,1]
        # ----------------------------------------------------
        imgs_final = imgs_final * 2 - 1
        imgs_final = torch.clamp(imgs_final, -1, 1)

        return imgs_final
