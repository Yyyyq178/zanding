import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import numpy as np
import os


class SRDataset(Dataset):
    def __init__(self, root, hr_size=256, lr_size=64, is_train=True, degradation_type='codeformer'):
        """
        Dataset：只负责读取 HR 图片并进行基础预处理。
        """
        self.dataset = ImageFolder(root)
        self.hr_size = hr_size
        self.lr_size = lr_size # 保留接口参数以兼容 main_mar.py 的调用
        self.is_train = is_train
        
        print(f"Dataset initialized (Lite Mode). Degradation '{degradation_type}' will be handled by the Engine on GPU.")

        # 归一化工具 (将 [0,1] 映射到 [-1, 1])
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __getitem__(self, index):
        # 1. 读取 HR 图片
        img_hr, _ = self.dataset[index]

        # 2. 裁剪与增强 (HR)
        # 训练时：随机裁剪 + 随机翻转
        # 验证时：中心裁剪
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
        
        # 3. 处理 HR Tensor 并归一化
        hr_tensor = torch.from_numpy(np.ascontiguousarray(img_hr_np.transpose(2, 0, 1))).float()
        hr_tensor = self.normalize(hr_tensor)

        # 4. LR Tensor 占位符
        # 为了保持 DataLoader 返回格式 (hr, lr, filename) 不变，
        # 我们返回一个与 HR 形状相同的全零 Tensor。
        # 训练循环(Engine)收到这个 tensor 后会立即用 GPU 生成的真实 LR 覆盖它。
        lr_tensor = torch.zeros_like(hr_tensor)

        # 获取文件名
        path, _ = self.dataset.samples[index]
        filename = os.path.basename(path)
        
        return hr_tensor, lr_tensor, filename 

    def __len__(self):
        return len(self.dataset)