import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PairedSRDataset(Dataset):
    """同时读取 HR 和 LR 文件夹 (用于 paired_test)"""
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

        img_hr = img_hr.resize((self.img_size, self.img_size), Image.BICUBIC)
        img_lr = img_lr.resize((self.img_size, self.img_size), Image.BICUBIC)

        return self.normalize(img_hr), self.normalize(img_lr), self.hr_files[index]

    def __len__(self):
        return len(self.hr_files)


class HROnlyDataset(Dataset):
    """[新增] 只读取 HR 文件夹 (用于自动退化测试)，支持扁平目录结构"""
    def __init__(self, root_hr, img_size=512):
        super().__init__()
        self.root_hr = root_hr
        self.img_size = img_size
        
        # 读取所有支持的图片格式
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
        
        # 强制 Resize 到 512，保证 VAE 不会因为尺寸不对而报错
        if img_hr.size != (self.img_size, self.img_size):
            img_hr = img_hr.resize((self.img_size, self.img_size), Image.BICUBIC)
        
        hr_tensor = self.normalize(img_hr)
        
        # 返回：HR, Dummy_LR (用HR占位), Filename
        # 真正的 LR 会在 engine_mar.py 中通过 degradation_model 在线生成
        return hr_tensor, hr_tensor, self.hr_files[index]

    def __len__(self):
        return len(self.hr_files)