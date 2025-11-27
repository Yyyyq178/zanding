import os
import numpy as np
import random
import torch
import torchvision.datasets as datasets
#from torchvision.transforms.functional import hflip


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target
    

# class PairedImageFolder(datasets.ImageFolder):
#     def __init__(self, root_hr, root_lr, transform_hr=None, transform_lr=None):
#         # 初始化父类，指定 HR 文件夹为主目录
#         super().__init__(root_hr, transform=None)
#         self.root_hr = root_hr
#         self.root_lr = root_lr
#         self.transform_hr = transform_hr
#         self.transform_lr = transform_lr

#     def __getitem__(self, index):
#         # 1. 获取 HR 图片的路径 (path_hr) 和 标签 (target)
#         # 这是 ImageFolder 自带的功能
#         path_hr, target = self.samples[index]
        
#         # 2. 获取对应的 LR 图片路径
#         # 逻辑：将路径字符串中的 HR 根目录部分，替换为 LR 根目录部分
#         # 例如：/data/HR_image/img1.jpg -> /data/LR_image/img1.jpg
#         path_lr = path_hr.replace(self.root_hr, self.root_lr)
        
#         # 3. 加载图片 (使用父类自带的 loader，通常是 PIL loader)
#         sample_hr = self.loader(path_hr)
#         sample_lr = self.loader(path_lr)
        
#         # 4. 同步随机翻转 (50% 概率)
#         # 必须保证 HR 和 LR 要么都翻，要么都不翻
#         if random.random() > 0.5:
#             sample_hr = hflip(sample_hr)
#             sample_lr = hflip(sample_lr)
            
#         # 5. 应用各自的转换 (比如裁剪、归一化)
#         if self.transform_hr is not None:
#             sample_hr = self.transform_hr(sample_hr)
#         if self.transform_lr is not None:
#             sample_lr = self.transform_lr(sample_lr)

#         # 返回：HR图片, LR图片
#         # 注意：我们这里不再返回 label，因为超分任务通常不需要类别标签
#         return sample_hr, sample_lr