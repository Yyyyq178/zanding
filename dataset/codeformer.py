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

class CodeFormerDegradation:
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
        blur_kernel_size=41,
        kernel_list=("iso", "aniso", "generalized_iso", "generalized_aniso"),
        kernel_prob=(0.5, 0.5, 0.25, 0.25),
        blur_sigma=(0.1, 12),
        downsample_range=(1, 12),
        noise_range=(0, 15/255.0),
        jpeg_range=(30, 100),
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
        imgs_hr: [B,C,H,W], RGB, [-1,1] torch
        """
        device = imgs_hr.device
        b, c, h, w = imgs_hr.shape

        # ============================================================
        # 1. torch tensor RGB[-1,1] → numpy BGR[0,1]
        # ============================================================
        imgs_np = (imgs_hr.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2  # RGB→[0,1]
        imgs_np = imgs_np[..., ::-1]  # RGB→BGR
        imgs_np = imgs_np.astype(np.float32)

        imgs_lq_list = []

        for i in range(b):
            img = imgs_np[i]  # HWC, BGR, float32, [0,1]

            # --------------------------------------------
            # Step 1. Blur (cv2.filter2D)
            # --------------------------------------------
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

            # --------------------------------------------
            # Step 2. Downsample
            # --------------------------------------------
            # Random scale
            if scale is None:
                scale = random.uniform(*self.downsample_range)
            scale = float(scale)
            
            h_lr = int(h // scale)
            w_lr = int(w // scale)
            img = cv2.resize(img, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)

            # --------------------------------------------
            # Step 3. Gaussian Noise
            # --------------------------------------------
            img = random_add_gaussian_noise(img, self.noise_range)

            # --------------------------------------------
            # Step 4. JPEG
            # --------------------------------------------
            img = random_add_jpg_compression(img, self.jpeg_range)

            # --------------------------------------------
            # Step 5. Upsample back
            # --------------------------------------------
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            imgs_lq_list.append(img)

        # Stack
        imgs_lq = np.stack(imgs_lq_list, axis=0)  # [B,H,W,C]

        # ============================================================
        # 6. numpy BGR[0,1] → torch RGB[-1,1]
        # ============================================================
        imgs_lq = imgs_lq[..., ::-1]  # BGR→RGB
        imgs_lq = imgs_lq.transpose(0, 3, 1, 2)  # BHWC→BCHW
        imgs_lq = torch.from_numpy(imgs_lq.copy()).to(device)

        imgs_lq = imgs_lq * 2 - 1
        imgs_lq = torch.clamp(imgs_lq, -1, 1)

        return imgs_lq
