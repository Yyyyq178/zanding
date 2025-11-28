import zipfile
import os
import random
import shutil

# ================= 配置区域 =================
# 1. 压缩包路径 
# ⚠️ 请修改为你真实的文件名 (注意大小写!)
# 例如: "/root/autodl-fs/ffhq512.zip"
zip_file_path = "/root/autodl-fs/hlwu/FFHQ512.zip"

# 2. 解压目标根目录 (脚本会自动创建 HR_image 等子文件夹)
target_root = "/root/autodl-tmp/zanding/data"

# 3. 解压总数量
limit_count = 70000

# 4. 验证集比例 (0.1 表示 10% 做验证集)
val_ratio = 0.1
# ===========================================

def setup_dir(path):
    """安全创建目录"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    # 检查压缩包是否存在
    if not os.path.exists(zip_file_path):
        print(f"❌ 错误: 找不到文件 {zip_file_path}")
        print("   请修改脚本中的 zip_file_path 变量！")
        return

    # 1. 准备符合 ImageFolder 标准的目录结构
    # 结构: data/HR_image/train/images/xxx.jpg
    # 注意: 'images' 这一层子文件夹是必须的，它是 ImageFolder 识别的“类别名”
    train_dir = os.path.join(target_root, "HR_image", "train", "images")
    val_dir = os.path.join(target_root, "HR_image", "val", "images")
    
    setup_dir(train_dir)
    setup_dir(val_dir)
    
    print(f"📂 正在读取压缩包索引: {zip_file_path} ...")
    
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # 2. 获取并筛选图片文件
        all_files = z.namelist()
        # 过滤出常见的图片格式
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        # 排除 Mac 系统可能产生的隐藏文件 (._ 开头)
        image_files = [f for f in image_files if not os.path.basename(f).startswith("._")]
        
        # 3. 排序并截取
        image_files.sort() # 排序保证确定性
        
        if len(image_files) < limit_count:
            print(f"⚠️ 警告: 压缩包内只有 {len(image_files)} 张图片，不足 {limit_count} 张。将全部使用。")
            selected_files = image_files
        else:
            selected_files = image_files[:limit_count]
            
        print(f"📊 选中了 {len(selected_files)} 张图片用于解压。")
        
        # 4. 打乱并划分 Train/Val
        random.seed(42) # 固定随机种子，保证复现性
        random.shuffle(selected_files)
        
        val_count = int(len(selected_files) * val_ratio)
        val_files = selected_files[:val_count]
        train_files = selected_files[val_count:]
        
        print(f"   - 训练集: {len(train_files)} 张 -> 存入 {train_dir}")
        print(f"   - 验证集: {len(val_files)} 张 -> 存入 {val_dir}")
        
        # 5. 定义解压函数
        def extract_list(files, dest_dir):
            count = 0
            for file_path in files:
                try:
                    # 获取纯文件名 (去除压缩包内的文件夹路径，直接“平铺”到目标目录)
                    file_name = os.path.basename(file_path)
                    target_path = os.path.join(dest_dir, file_name)
                    
                    # 读取流并写入文件 (比先解压再移动更高效)
                    with z.open(file_path) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    
                    count += 1
                    if count % 200 == 0:
                        print(f"      已处理 {count}/{len(files)} ...", end='\r')
                except Exception as e:
                    print(f"\n❌ 解压 {file_path} 失败: {e}")
            print(f"\n      ✅ 完成！")

        # 6. 执行解压
        print("🚀 开始解压训练集...")
        extract_list(train_files, train_dir)
        
        print("🚀 开始解压验证集...")
        extract_list(val_files, val_dir)

    print("\n🎉 数据准备完成！")
    print(f"   HR 数据路径 (用于训练命令): {os.path.join(target_root, 'HR_image')}")

if __name__ == "__main__":
    main()