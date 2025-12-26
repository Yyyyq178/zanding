import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import numpy as np
import os

class SRDataset(Dataset):
    def __init__(self, root, hr_size=512, lr_size=None, is_train=True, degradation_type='codeformer'):
        """
        Dataset：只负责读取 HR 图片并进行基础预处理。
        """
        self.dataset = ImageFolder(root)
        self.hr_size = hr_size
        self.lr_size = lr_size 
        self.is_train = is_train
        
        mode = "Training" if is_train else "Validation"
        print(f"Dataset initialized ({mode}). HR Size: {hr_size}. Logic: >{hr_size} -> Crop, else -> Resize.")

        # 归一化工具 (将 [0,1] 映射到 [-1, 1])
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __getitem__(self, index):
        # 读取 HR 图片
        img_hr, _ = self.dataset[index]

        # 裁剪与增强 (HR)
        # 逻辑：无论什么阶段，只要大于目标尺寸，就裁剪。
        W, H = img_hr.size
        
        if self.is_train:
            # --- 训练阶段：随机裁剪 ---
            if W > self.hr_size and H > self.hr_size:
                # 图片够大，随机裁剪
                i, j, h, w = transforms.RandomCrop.get_params(
                    img_hr, output_size=(self.hr_size, self.hr_size)
                )
                img_hr = TF.crop(img_hr, i, j, h, w)
            else:
                # 图片不够大，强制缩放
                img_hr = TF.resize(img_hr, (self.hr_size, self.hr_size))
            
            # 训练时增加随机翻转
            if random.random() > 0.5:
                img_hr = TF.hflip(img_hr)
        else:
            # --- 验证阶段：中心裁剪 (方便看指标) ---
            if W > self.hr_size and H > self.hr_size:
                # 图片够大，中心裁剪
                img_hr = TF.center_crop(img_hr, (self.hr_size, self.hr_size))
            else:
                # 图片不够大，强制缩放
                img_hr = TF.resize(img_hr, (self.hr_size, self.hr_size))

        # 转换: PIL -> Numpy (H, W, C) [0, 1] RGB, float32
        img_hr_np = np.array(img_hr).astype(np.float32) / 255.0
        
        # 处理 HR Tensor 并归一化
        hr_tensor = torch.from_numpy(np.ascontiguousarray(img_hr_np.transpose(2, 0, 1))).float()
        hr_tensor = self.normalize(hr_tensor)

        # LR Tensor 占位符 (由 Engine 在 GPU 上生成)
        lr_tensor = torch.zeros_like(hr_tensor)

        # 获取文件名
        path, _ = self.dataset.samples[index]
        filename = os.path.basename(path)
        
        return hr_tensor, lr_tensor, filename 

    def __len__(self):
        return len(self.dataset)