import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from dataset.degradation import random_add_gaussian_noise_pt

class CodeFormerDegradation:
    """
    CodeFormer 退化模拟器
    将退化流程封装为类，便于统一管理超参数。
    流程: HR -> [Random Blur] -> Downsample -> [Random Noise] -> Upsample -> Output
    """
    
    def __init__(
        self, 
        blur_prob=0.5, 
        blur_sigma_range=(0.1, 12.0), 
        blur_kernel_size=41,
        downsample_scale_range=(2.0, 4.0),
        noise_prob=0.5,
        noise_sigma_range=(0, 15.0/255.0),
        noise_gray_prob=0.0
    ):
        """
        在初始化函数中统一定义所有变量
        """
        # === 1. 模糊相关参数 ===
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.blur_kernel_size = blur_kernel_size
        
        # === 2. 下采样相关参数 ===
        self.downsample_scale_range = downsample_scale_range
        
        # === 3. 噪声相关参数 ===
        self.noise_prob = noise_prob
        self.noise_sigma_range = noise_sigma_range
        self.noise_gray_prob = noise_gray_prob

    def __call__(self, imgs_hr, scale=None):
        """
        执行退化逻辑
        Args:
            imgs_hr: [B, C, H, W], [-1, 1]
            scale: 指定下采样倍率，如果为None则使用 self.downsample_scale_range 中的随机值
        """
        # 1. 准备工作：Detach 并反归一化到 [0, 1]
        b, c, h, w = imgs_hr.shape
        imgs = (imgs_hr.detach() + 1.0) * 0.5  
        
        # 2. 确定下采样倍率
        if scale is None:
            # 使用初始化时定义的范围
            s = random.uniform(self.downsample_scale_range[0], self.downsample_scale_range[1])
        else:
            s = scale
            
        h_lr = int(h // s)
        w_lr = int(w // s)

        # =======================================
        # Stage 1: Random Gaussian Blur
        # =======================================
        if random.random() < self.blur_prob:
            sigma = random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])
            imgs = TF.gaussian_blur(imgs, kernel_size=self.blur_kernel_size, sigma=[sigma, sigma])

        # =======================================
        # Stage 2: Downsampling (Bicubic)
        # =======================================
        imgs_lr = F.interpolate(
            imgs, 
            size=(h_lr, w_lr), 
            mode='bilinear', 
            align_corners=False, 
            antialias=True
        )

        # =======================================
        # Stage 3: Random Gaussian Noise
        # =======================================
        if random.random() < self.noise_prob:
            imgs_lr = random_add_gaussian_noise_pt(
                imgs_lr,
                sigma_range=self.noise_sigma_range,
                clip=True,
                rounds=False, 
                gray_prob=self.noise_gray_prob
            )

        # =======================================
        # Stage 4: Upsampling (Back to HR size)
        # =======================================
        imgs_final = F.interpolate(
            imgs_lr, 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False, 
            antialias=True
        )

        # 3. 归一化回 [-1, 1] 并截断
        imgs_final = (imgs_final - 0.5) * 2.0
        imgs_final = torch.clamp(imgs_final, -1.0, 1.0)
        
        return imgs_final