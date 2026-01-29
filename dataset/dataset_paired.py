import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import numpy as np

class PairedSRDataset(Dataset):
    """
    用于 Paired Test (HR + LR 文件夹)。
    逻辑：测试阶段 -> 随机裁剪 (HR 和 LR 保持同步裁剪)
    """
    def __init__(self, root_hr, root_lr, img_size=512):
        super().__init__()
        self.root_hr = root_hr
        self.root_lr = root_lr
        self.img_size = img_size
        
        self.hr_files = sorted([f for f in os.listdir(root_hr) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        self.lr_files = sorted([f for f in os.listdir(root_lr) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        
        min_len = min(len(self.hr_files), len(self.lr_files))
        self.hr_files = self.hr_files[:min_len]
        self.lr_files = self.lr_files[:min_len]

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        hr_path = os.path.join(self.root_hr, self.hr_files[index])
        lr_path = os.path.join(self.root_lr, self.lr_files[index])
        
        img_hr = Image.open(hr_path).convert('RGB')
        img_lr = Image.open(lr_path).convert('RGB')

        W, H = img_hr.size
        # 如果需要 resize 
        if not (W > self.img_size and H > self.img_size):
            # PIL -> Numpy
            img_hr = np.array(img_hr)
            img_lr = np.array(img_lr)
            
            # 使用 cv2.resize 
            img_hr = cv2.resize(img_hr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_lr = cv2.resize(img_lr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            
            # Numpy -> Tensor 
            img_hr = Image.fromarray(img_hr)
            img_lr = Image.fromarray(img_lr)
            
        else:
             i, j, h, w = transforms.RandomCrop.get_params(
                img_hr, output_size=(self.img_size, self.img_size))
             img_hr = TF.crop(img_hr, i, j, h, w)
             img_lr = TF.crop(img_lr, i, j, h, w)

        return self.normalize(img_hr), self.normalize(img_lr), self.hr_files[index]

    def __len__(self):
        return len(self.hr_files)


class HROnlyDataset(Dataset):
    """
    用于 HR Only Test (自动退化)。
    逻辑：测试阶段 -> 随机裁剪 HR
    """
    def __init__(self, root_hr, img_size=512):
        super().__init__()
        self.root_hr = root_hr
        self.img_size = img_size
        
        self.hr_files = sorted([
            f for f in os.listdir(root_hr) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
        ])
        
        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No images found in {root_hr}")

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        hr_path = os.path.join(self.root_hr, self.hr_files[index])
        img_hr = Image.open(hr_path).convert('RGB')
        
        # --- 修改为随机裁剪 ---
        W, H = img_hr.size
        
        if W > self.img_size and H > self.img_size:
            # 随机裁剪
            i, j, h, w = transforms.RandomCrop.get_params(img_hr, output_size=(self.img_size, self.img_size))
            img_hr = TF.crop(img_hr, i, j, h, w)
        else:
            # 缩放
            img_hr = TF.resize(img_hr, (self.img_size, self.img_size))
        
        hr_tensor = self.normalize(img_hr)
        
        return hr_tensor, hr_tensor, self.hr_files[index]

    def __len__(self):
        return len(self.hr_files)