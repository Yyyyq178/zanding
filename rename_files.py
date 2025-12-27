import os

# ================= 配置区域 =================
# 请在这里填入你需要修改图片名字的文件夹路径
target_folder = '/root/autodl-tmp/zanding/output_zanding_order/ariter64-diffstepsddim100-temp1.0-linearcfg1.0-image_num70000_ema_evaluate/sr_images'
# ===========================================

def remove_prefix():
    # 1. 检查文件夹是否存在
    if not os.path.exists(target_folder):
        print(f"错误：找不到文件夹路径：{target_folder}")
        return

    print(f"正在扫描文件夹: {target_folder} ...")
    
    count = 0
    # 获取文件夹下的所有文件
    file_list = os.listdir(target_folder)
    
    for filename in file_list:
        # 只处理常见的图片格式
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            continue
            
        # 分离文件名和后缀。例如 'rank0_6.png' -> name='rank0_6', ext='.png'
        name, ext = os.path.splitext(filename)
        
        # --- 修改部分开始 ---
        # 核心逻辑：检查文件名是否以 'rank0_' 开头
        if name.startswith('rank0_'):
            # 去掉前6个字符（即去掉 'rank0_'，因为 rank0_ 长度为6）
            new_name_stem = name[6:]
            
            # 组合新名字。例如 '6' + '.png' -> '6.png'
            new_filename = new_name_stem + ext
            
            old_path = os.path.join(target_folder, filename)
            new_path = os.path.join(target_folder, new_filename)
            
            # 安全检查：防止新名字的文件已经存在（避免覆盖）
            if os.path.exists(new_path):
                print(f"[跳过] 目标文件已存在: {new_filename}，跳过重命名 {filename}")
                continue
            
            # 执行重命名
            os.rename(old_path, new_path)
            print(f"[成功] {filename} -> {new_filename}")
            count += 1
        # --- 修改部分结束 ---
            
    print("-" * 30)
    print(f"全部完成！共重命名了 {count} 张图片。")

if __name__ == '__main__':
    remove_prefix()