import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import random

class SRDataset(Dataset):
    def __init__(self, root, hr_size=128, lr_size=32, is_train=True):
        """
        root: 数据集根目录 (例如 /data/HR_image/train)
        hr_size: 裁剪后的 HR 大小 (128)
        lr_size: 下采样后的 LR 大小 (32)
        is_train: 训练集开启随机裁剪，验证集使用中心裁剪
        """
        # 利用 ImageFolder 自动处理目录结构读取图片
        self.dataset = ImageFolder(root)
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.is_train = is_train
        
        # 定义归一化 (MAR 使用 0.5 均值)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        # 1. 读取原始大图 (PIL Image)
        img_hr, _ = self.dataset[index]

        # 2. 裁剪与增强 (HR)
        if self.is_train:
            # --- 随机裁剪 (Random Crop) ---
            # get_params 会返回随机生成的坐标 (i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(
                img_hr, output_size=(self.hr_size, self.hr_size)
            )
            img_hr = TF.crop(img_hr, i, j, h, w)

            # --- 随机翻转 (Random Flip) ---
            if random.random() > 0.5:
                img_hr = TF.hflip(img_hr)
        else:
            # 验证集使用中心裁剪，保证结果确定性
            img_hr = TF.center_crop(img_hr, (self.hr_size, self.hr_size))

        # 3. 生成 LR (下采样)
        # 使用裁剪好的 HR 生成对应的 LR
        img_lr = TF.resize(
            img_hr, 
            size=[self.lr_size, self.lr_size], 
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        # 4. 转 Tensor 并归一化
        hr_tensor = self.norm(img_hr)
        lr_tensor = self.norm(img_lr)

        # 返回一对图片 (HR, LR)
        return hr_tensor, lr_tensor

    def __len__(self):
        return len(self.dataset)