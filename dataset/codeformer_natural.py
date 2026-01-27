import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2

from dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)

class CodeFormerDegradationNatural:
    """
    High-level behavior aligned with CodeFormer Dataset pipeline:
    HR[-1,1] (RGB) 
        → convert to BGR [0,1] numpy
        → blur (cv2.filter2D)
        → downsample (cv2.resize)
        → noise
        → JPEG
        → upsample (cv2.resize)
        → back to RGB[-1,1] torch
    """

    def __init__(
        self,
        blur_kernel_size=21, 
        kernel_list=("iso", "aniso"), 
        kernel_prob=(0.5, 0.5),
        blur_sigma=(0.2, 3), 
        downsample_range=(4, 4), 
        noise_range=(1, 12),
        jpeg_range=(60, 100),
    ):
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    @torch.no_grad()
    def __call__(self, imgs_hr, scale=None):
        """
        处理流程与原版一致，只是参数不同。
        """
        device = imgs_hr.device
        b, c, h, w = imgs_hr.shape

        # RGB[-1,1] -> BGR[0,1]
        imgs_np = (imgs_hr.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2
        imgs_np = imgs_np[..., ::-1].astype(np.float32)

        imgs_lq_list = []

        for i in range(b):
            img = imgs_np[i]

            # 1. Blur
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-np.pi, np.pi],
                noise_range=None,
            )
            img = cv2.filter2D(img, -1, kernel)

            # 2. Downsample
            if scale is None:
                # 这里 random.uniform(4, 4) 会直接返回 4.0
                scale = random.uniform(*self.downsample_range)
            scale = float(scale)
            
            h_lr = int(h // scale)
            w_lr = int(w // scale)
            img = cv2.resize(img, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)

            # 3. Noise
            img = random_add_gaussian_noise(img, self.noise_range)

            # 4. JPEG
            img = random_add_jpg_compression(img, self.jpeg_range)

            # 5. Upsample back
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            imgs_lq_list.append(img)

        # Stack & Convert back
        imgs_lq = np.stack(imgs_lq_list, axis=0)
        imgs_lq = imgs_lq[..., ::-1].transpose(0, 3, 1, 2)
        imgs_lq = torch.from_numpy(imgs_lq.copy()).to(device)

        imgs_lq = imgs_lq * 2 - 1
        imgs_lq = torch.clamp(imgs_lq, -1, 1)

        return imgs_lq