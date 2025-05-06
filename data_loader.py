import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms
import re


class MedicalImageDataset(Dataset):
    """
    医学图像配准数据集，用于加载CT、MRI和形变MRI图像三元组
    以灰度图像形式加载
    """

    def __init__(self, root_dir, is_train=True, transform=None):
        """
        参数:
            root_dir (str): 数据集根目录路径
            is_train (bool): 是否为训练集
            transform (callable, optional): 可选的数据变换
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform

        # 确定数据集所在文件夹（训练集或测试集）
        self.data_folder = "train" if is_train else "test"
        self.data_path = os.path.join(root_dir, self.data_folder)

        # CT文件夹
        self.ct_folder = os.path.join(self.data_path, "CT")
        # 原始MRI文件夹
        self.mri_folder = os.path.join(self.data_path, "MRI")
        # 形变后的MRI文件夹
        self.mri_deformed_folder = os.path.join(self.data_path, "MRI_deformed")

        # 获取CT图像列表
        self.ct_files = sorted([f for f in os.listdir(self.ct_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 获取MRI图像列表
        self.mri_files = sorted([f for f in os.listdir(self.mri_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 构建CT-MRI-形变MRI的三元组关系
        self.sample_triplets = self._build_sample_triplets()

    def _build_sample_triplets(self):
        """
        构建CT-MRI-形变MRI的三元组关系
        """
        sample_triplets = []

        # 获取所有形变后的MRI文件
        deformed_files = sorted([f for f in os.listdir(self.mri_deformed_folder)
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])

        print(f"找到 {len(deformed_files)} 个形变MRI图像")

        # 使用正则表达式解析形变MRI文件名
        pattern = r'(\d+)_deformed_\d+\.(?:png|jpg|jpeg)$'

        # 遍历所有形变后的MRI文件，找到对应的CT和MRI文件
        for deformed_file in deformed_files:
            match = re.match(pattern, deformed_file)
            if not match:
                print(f"警告：无法从文件名 {deformed_file} 解析出原始MRI索引，跳过")
                continue

            # 提取基本文件名（不包括扩展名）
            base_name = match.group(1)

            # 查找对应的CT和MRI文件
            ct_file = None
            mri_file = None

            # 遍历CT文件列表，查找匹配的文件名
            for ct in self.ct_files:
                ct_base = os.path.splitext(ct)[0]
                if ct_base == base_name:
                    ct_file = ct
                    break

            # 遍历MRI文件列表，查找匹配的文件名
            for mri in self.mri_files:
                mri_base = os.path.splitext(mri)[0]
                if mri_base == base_name:
                    mri_file = mri
                    break

            # 如果找到匹配的CT和MRI文件，则添加到三元组列表中
            if ct_file and mri_file:
                # 输出配对信息，用于调试
                if len(sample_triplets) < 5 or len(sample_triplets) % 50 == 0:
                    print(f"配对样本 {len(sample_triplets)}: CT={ct_file}, MRI={mri_file}, 形变MRI={deformed_file}")

                sample_triplets.append({
                    'ct': ct_file,
                    'mri': mri_file,
                    'deformed_mri': deformed_file
                })
            else:
                print(f"警告：形变MRI {deformed_file} 没有找到对应的CT或MRI文件，跳过")

        print(f"成功构建 {len(sample_triplets)} 个有效样本三元组")
        return sample_triplets

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.sample_triplets)

    def __getitem__(self, idx):
        """
        获取一个样本
        """
        # 获取当前样本三元组
        triplet = self.sample_triplets[idx]

        # 读取CT图像为灰度图像
        ct_path = os.path.join(self.ct_folder, triplet['ct'])
        ct_image = Image.open(ct_path).convert('L')  # 'L'表示灰度图像

        # 读取原始MRI图像为灰度图像
        mri_path = os.path.join(self.mri_folder, triplet['mri'])
        mri_image = Image.open(mri_path).convert('L')

        # 读取形变后的MRI图像为灰度图像
        deformed_mri_path = os.path.join(self.mri_deformed_folder, triplet['deformed_mri'])
        deformed_mri_image = Image.open(deformed_mri_path).convert('L')

        # 应用变换
        if self.transform:
            ct_image = self.transform(ct_image)
            mri_image = self.transform(mri_image)
            deformed_mri_image = self.transform(deformed_mri_image)

        # 返回文件名作为元组
        filenames = (triplet['ct'], triplet['mri'], triplet['deformed_mri'])

        return {
            'ct': ct_image,
            'mri': mri_image,
            'deformed_mri': deformed_mri_image,
            'filenames': filenames
        }


def get_data_loaders(root_dir, batch_size=4, num_workers=4):
    """
    创建数据加载器

    参数:
        root_dir (str): 数据集根目录
        batch_size (int): 批处理大小
        num_workers (int): 数据加载的工作线程数

    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 定义数据变换 (适用于灰度图像)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道灰度图像
    ])

    # 创建训练数据集
    train_dataset = MedicalImageDataset(
        root_dir=root_dir,
        is_train=True,
        transform=transform
    )

    # 创建测试数据集
    test_dataset = MedicalImageDataset(
        root_dir=root_dir,
        is_train=False,
        transform=transform
    )

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时不打乱数据，保持顺序
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader