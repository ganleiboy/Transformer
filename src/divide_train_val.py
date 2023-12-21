# Python编程：一个目录A中有100个子目录，每个子目录中有600张图像。
# 需求：将目录A中每个子目录都按照8:2的比例进行划分，按照ImageNet的格式构建训练集train和验证集val。

import os
import random
import shutil
from tqdm import tqdm

# 设置目录A的路径
directory_A = "/home/pd_mzc/Documents/ViT/data/miniImagenet/data"

# 创建训练集和验证集目录
train_dir = "/home/pd_mzc/Documents/ViT/data/miniImagenet/train"
val_dir = "/home/pd_mzc/Documents/ViT/data/miniImagenet/val"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历目录A中的每个子目录
for subdir in tqdm(os.listdir(directory_A)):
    subdir_path = os.path.join(directory_A, subdir)
    if os.path.isdir(subdir_path):
        # 创建训练集和验证集的子目录
        train_subdir = os.path.join(train_dir, subdir)
        val_subdir = os.path.join(val_dir, subdir)
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(val_subdir, exist_ok=True)
        
        # 获取子目录中的图像文件列表
        images = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        # 随机打乱图像文件列表
        random.shuffle(images)
        
        # 计算划分比例
        num_images = len(images)
        num_train = int(num_images * 0.8)
        
        # 将图像文件移动到训练集或验证集目录
        for i, image in enumerate(images):
            src_path = os.path.join(subdir_path, image)
            if i < num_train:
                dst_path = os.path.join(train_subdir, image)
            else:
                dst_path = os.path.join(val_subdir, image)
            shutil.move(src_path, dst_path)
print("+"*30 + " end " + "+"*30)