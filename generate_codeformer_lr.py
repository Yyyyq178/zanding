import os
import cv2
import numpy as np
import argparse
import random
import multiprocessing
from tqdm import tqdm
from functools import partial

# 尝试导入项目中的退化工具
try:
    from dataset.degradation import (
        random_mixed_kernels,
        random_add_gaussian_noise,
        random_add_jpg_compression,
    )
except ImportError:
    print("错误: 无法导入 dataset.degradation。")
    print("请确保将此脚本放在项目根目录下（即与 dataset/ 文件夹同级），并且 dataset/degradation.py 存在。")
    exit(1)

def process_one_image(file_info, args):
    """
    处理单张图片的函数
    逻辑严格对齐 dataset/codeformer_face.py
    """
    img_path, save_path = file_info
    
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Warning] 无法读取图片: {img_path}")
        return

    # ============================================================
    # 预处理：强制 HR 为 512x512
    # ============================================================
    # 为了保证输出严格是 512x512，且退化逻辑（Kernel大小等）与 CodeFormer 训练时一致，
    # 我们必须在退化开始前将图片固定为 512x512。
    # 如果原图不是 512x512，这里会进行调整。
    target_size = 512
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # 转换为 float32 [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # 记录当前 HR 的尺寸 (此时已经是 512x512)
    h, w, _ = img.shape

    # ============================================================
    # 复刻 CodeFormerDegradation 的核心逻辑
    # ============================================================

    # --------------------------------------------
    # Step 1. 混合模糊 (Blur)
    # --------------------------------------------
    # 参数源自 CodeFormerDegradation.__init__ 默认值
    kernel = random_mixed_kernels(
        kernel_list=("iso", "aniso", "generalized_iso", "generalized_aniso"),
        kernel_prob=(0.5, 0.5, 0.25, 0.25),
        kernel_size=41,
        sigma_x_range=(0.1, 15),
        sigma_y_range=(0.1, 15),
        rotation_range=[-np.pi, np.pi],
        noise_range=None
    )
    img = cv2.filter2D(img, -1, kernel)

    # --------------------------------------------
    # Step 2. 下采样 (Downsample)
    # --------------------------------------------
    # 严格对齐 CodeFormer: scale = random.uniform(0.8, 32)
    scale = random.uniform(0.8, 32)
    
    h_lr = int(h // scale)
    w_lr = int(w // scale)
    
    # 保护措施：cv2.resize 目标尺寸不能为 0
    # 虽然 CodeFormer 源码没显式写，但在极端的 scale=32 时，512/32=16，是安全的。
    # 只要不小于 1 即可。
    h_lr = max(h_lr, 1)
    w_lr = max(w_lr, 1)
    
    # 严格对齐: 使用 cv2.INTER_LINEAR
    img = cv2.resize(img, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)

    # --------------------------------------------
    # Step 3. 高斯噪声 (Gaussian Noise)
    # --------------------------------------------
    # 严格对齐: noise_range=(0, 20)
    img = random_add_gaussian_noise(img, sigma_range=(0, 20))

    # --------------------------------------------
    # Step 4. JPEG 压缩 (JPEG Compression)
    # --------------------------------------------
    # 严格对齐: jpeg_range=(30, 100)
    img = random_add_jpg_compression(img, quality_range=(30, 100))

    # --------------------------------------------
    # Step 5. 上采样 (Upsample back)
    # --------------------------------------------
    # 严格对齐: resize 回原图大小 (512x512)，使用 INTER_LINEAR
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # ============================================================
    
    # 转换回 uint8 [0, 255] 并保存
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description="生成 CodeFormer LR 数据集 (固定输出 512x512)")
    parser.add_argument('--input', type=str, required=True, help='输入 HR 图片文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出 LR 图片文件夹路径')
    parser.add_argument('--num_workers', type=int, default=8, help='并行处理的进程数')
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件夹不存在 -> {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)

    # 收集图片文件
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif'}
    tasks = []
    
    print(f"正在扫描文件夹: {args.input} ...")
    for root, dirs, files in os.walk(args.input):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in exts:
                input_path = os.path.join(root, filename)
                
                # 计算输出路径，保持子文件夹结构
                rel_path = os.path.relpath(input_path, args.input)
                output_path = os.path.join(args.output, rel_path)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                tasks.append((input_path, output_path))

    total = len(tasks)
    print(f"找到 {total} 张图片。")
    print("配置: 输出尺寸=512x512, Downsample=(0.8, 32), Interpolation=LINEAR")
    
    # 多进程处理
    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            func = partial(process_one_image, args=args)
            list(tqdm(pool.imap_unordered(func, tasks), total=total))
    else:
        for task in tqdm(tasks):
            process_one_image(task, args)

    print(f"\n处理完成！图片已保存至: {args.output}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()