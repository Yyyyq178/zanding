import os
import cv2
import numpy as np
import argparse
import random
import multiprocessing
from tqdm import tqdm
from functools import partial

# 导入项目中现有的退化函数
# 确保此脚本位于项目根目录，且 dataset 文件夹存在
try:
    from dataset.degradation import (
        random_mixed_kernels,
        random_add_gaussian_noise,
        random_add_jpg_compression,
    )
except ImportError:
    print("错误: 无法导入 dataset.degradation。请确保此脚本位于项目根目录，并且 dataset/ 文件夹完整。")
    exit(1)

def process_one_image(file_info, args):
    """
    处理单张图片的函数
    """
    img_path, save_path = file_info
    
    # 1. 读取图片 (BGR, uint8)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Cannot read {img_path}")
        return

    # 转换为 float32 [0, 1]
    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # =========================================================
    # 复刻 CodeFormerDegradation 的逻辑
    # =========================================================
    
    # --- Step 1. 混合模糊 (Blur) ---
    # 参数参考 dataset/codeformer.py 的默认值
    kernel = random_mixed_kernels(
        kernel_list=("iso", "aniso", "generalized_iso", "generalized_aniso"),
        kernel_prob=(0.5, 0.5, 0.25, 0.25),
        kernel_size=41,
        sigma_x_range=(0.1, 12), # blur_sigma
        sigma_y_range=(0.1, 12),
        rotation_range=[-np.pi, np.pi],
        noise_range=None
    )
    img = cv2.filter2D(img, -1, kernel)

    # --- Step 2. 下采样 (Downsample) ---
    # 随机选择 1 到 12 倍的下采样率
    scale = random.uniform(1, 12)
    h_lr = int(h // scale)
    w_lr = int(w // scale)
    
    # 防止尺寸过小 (至少 8x8)
    h_lr = max(h_lr, 8)
    w_lr = max(w_lr, 8)
    
    img = cv2.resize(img, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)

    # --- Step 3. 高斯噪声 (Gaussian Noise) ---
    # noise_range=(0, 15) -> 对应 dataset/codeformer.py
    img = random_add_gaussian_noise(img, sigma_range=(0, 15))

    # --- Step 4. JPEG 压缩 ---
    # jpeg_range=(30, 100) -> 对应 dataset/codeformer.py
    img = random_add_jpg_compression(img, quality_range=(30, 100))

    # --- Step 5. (可选) 上采样回原尺寸 ---
    if args.output_mode == 'same':
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # =========================================================
    
    # 转换为 uint8 [0, 255] 并保存
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description="使用 CodeFormer 退化逻辑生成 LR 数据集")
    parser.add_argument('--input_dir', type=str, required=True, help='HR 图片输入文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='LR 图片输出文件夹路径')
    parser.add_argument('--output_mode', type=str, default='small', choices=['small', 'same'],
                        help='输出模式: small=保留小尺寸(推荐), same=还原回原尺寸(模糊)')
    parser.add_argument('--num_workers', type=int, default=8, help='多进程并行数量')
    args = parser.parse_args()

    # 检查输入路径
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集所有图片文件
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif'}
    tasks = []
    
    print(f"Scanning files in {args.input_dir}...")
    for root, dirs, files in os.walk(args.input_dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in img_extensions:
                input_path = os.path.join(root, filename)
                
                # 保持相对路径结构
                rel_path = os.path.relpath(input_path, args.input_dir)
                output_path = os.path.join(args.output_dir, rel_path)
                
                # 确保输出子文件夹存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                tasks.append((input_path, output_path))

    total_files = len(tasks)
    print(f"Found {total_files} images. Starting processing with {args.num_workers} workers...")
    print(f"Degradation: Blur -> Downsample(1-12x) -> Noise -> JPEG -> {'Upsample' if args.output_mode=='same' else 'Keep Small'}")

    # 使用多进程加速
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # 使用 partial 传递 args 参数
        func = partial(process_one_image, args=args)
        list(tqdm(pool.imap_unordered(func, tasks), total=total_files))

    print(f"\nProcessing complete. Saved to {args.output_dir}")

if __name__ == '__main__':
    # 修复 Windows 下多进程启动问题
    multiprocessing.freeze_support()
    main()