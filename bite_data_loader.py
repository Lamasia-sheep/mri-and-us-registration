"""
BITE 数据集加载器
=================
BITE (Brain Images of Tumors for Evaluation) 数据集 Group2
包含14个病人的 MRI 和术中超声 (iUS) 配对切片，已在 Talairach 空间中对齐。

数据结构:
  group2 - png/
    01/
      01_mr_tal_png/    (378张MRI切片)
      01a_us_tal_png/   (数量不等的US切片)
    02/ ...
    ...
    14/ ...

用途:
  - 预训练: 利用真实临床 MR-US 配对数据预训练配准网络
  - 验证: 在真实数据上评估配准模型的泛化能力
"""

import os
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms


class BITEDataset(Dataset):
    """
    BITE 数据集，加载 MRI-US 配对切片用于配准任务。

    对于每个病人，MR 和 US 切片通过相同的切片索引进行配对。
    仅保留 MR 和 US 都有足够有效内容（非黑区域 > 阈值）的切片对。

    配准方向: US (固定图像) ← MR (移动图像)
    即: 模型学习将术前 MR 变形配准到术中 US
    """

    def __init__(self, root_dir, patient_ids=None, transform=None,
                 content_threshold=0.10, intensity_threshold=15):
        """
        参数:
            root_dir (str): BITE 数据集根目录 (group2 - png 文件夹路径)
            patient_ids (list): 指定使用的病人编号列表，如 ['01','02',...]; None 表示全部
            transform (callable): 图像变换
            content_threshold (float): 有效内容比例阈值 (默认10%)
            intensity_threshold (int): 像素强度阈值，低于此值视为背景
        """
        self.root_dir = root_dir
        self.transform = transform
        self.content_threshold = content_threshold
        self.intensity_threshold = intensity_threshold

        # 获取所有病人目录
        all_patients = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if patient_ids is not None:
            all_patients = [p for p in all_patients if p in patient_ids]

        # 构建所有有效的 MR-US 配对
        self.pairs = []
        self.patient_info = {}

        for pid in all_patients:
            pdir = os.path.join(root_dir, pid)
            subdirs = sorted(os.listdir(pdir))

            # 找到 MR 和 US 子目录
            mr_dirs = [s for s in subdirs if 'mr' in s.lower() and os.path.isdir(os.path.join(pdir, s))]
            us_dirs = [s for s in subdirs if 'us' in s.lower() and os.path.isdir(os.path.join(pdir, s))]

            if not mr_dirs or not us_dirs:
                print(f"警告: 病人 {pid} 缺少 MR 或 US 目录，跳过")
                continue

            mr_dir = os.path.join(pdir, mr_dirs[0])
            us_dir = os.path.join(pdir, us_dirs[0])

            # 获取切片文件
            mr_files = {int(f.replace('.png', '')): f
                        for f in os.listdir(mr_dir) if f.endswith('.png')}
            us_files = {int(f.replace('.png', '')): f
                        for f in os.listdir(us_dir) if f.endswith('.png')}

            # 找到共有的切片索引
            common_indices = sorted(set(mr_files.keys()) & set(us_files.keys()))

            # 筛选有效配对
            valid_count = 0
            for idx in common_indices:
                mr_path = os.path.join(mr_dir, mr_files[idx])
                us_path = os.path.join(us_dir, us_files[idx])

                # 快速检查内容有效性
                if self._is_valid_pair(mr_path, us_path):
                    self.pairs.append({
                        'patient_id': pid,
                        'slice_idx': idx,
                        'mr_path': mr_path,
                        'us_path': us_path
                    })
                    valid_count += 1

            self.patient_info[pid] = {
                'total_common': len(common_indices),
                'valid_pairs': valid_count,
                'mr_dir': mr_dirs[0],
                'us_dir': us_dirs[0]
            }

        print(f"\n===== BITE 数据集加载完成 =====")
        print(f"病人数: {len(self.patient_info)}")
        print(f"有效配对总数: {len(self.pairs)}")
        for pid, info in self.patient_info.items():
            print(f"  病人 {pid}: {info['valid_pairs']}/{info['total_common']} 有效配对")
        print(f"================================\n")

    def _is_valid_pair(self, mr_path, us_path):
        """检查 MR-US 配对是否都有足够的有效内容"""
        try:
            mr_img = np.array(Image.open(mr_path).convert('L'))
            us_img = np.array(Image.open(us_path).convert('L'))

            mr_ratio = (mr_img > self.intensity_threshold).sum() / mr_img.size
            us_ratio = (us_img > self.intensity_threshold).sum() / us_img.size

            return mr_ratio > self.content_threshold and us_ratio > self.content_threshold
        except Exception as e:
            print(f"警告: 无法读取图像 {mr_path} 或 {us_path}: {e}")
            return False

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        返回一个 MR-US 配对样本。

        配准方向: MR (fixed) ← US (moving)
        理由: MRI 解剖结构清晰、视野完整，作为固定图像能提供更好的监督信号。

        返回字典:
            'fixed': MR 图像 (固定图像/目标)
            'moving': US 图像 (移动图像/待配准)
            'patient_id': 病人编号
            'slice_idx': 切片索引
        """
        pair = self.pairs[idx]

        # 读取为灰度图像
        mr_image = Image.open(pair['mr_path']).convert('L')
        us_image = Image.open(pair['us_path']).convert('L')

        # 应用变换
        if self.transform:
            mr_image = self.transform(mr_image)
            us_image = self.transform(us_image)

        return {
            'fixed': mr_image,     # MRI 作为固定图像 (解剖清晰, 全脑覆盖)
            'moving': us_image,    # US 作为移动图像 (待配准到 MRI 空间)
            'patient_id': pair['patient_id'],
            'slice_idx': pair['slice_idx']
        }


def get_bite_data_loaders(root_dir, train_patients=None, val_patients=None,
                          batch_size=16, num_workers=0):
    """
    创建 BITE 数据集的训练和验证数据加载器。

    默认按病人划分: 前11个病人用于训练, 后3个用于验证。

    参数:
        root_dir (str): BITE 数据集根目录
        train_patients (list): 训练集病人编号; None 使用默认划分
        val_patients (list): 验证集病人编号; None 使用默认划分
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数

    返回:
        train_loader, val_loader
    """
    # 默认按病人划分: 11个训练, 3个验证
    if train_patients is None:
        train_patients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    if val_patients is None:
        val_patients = ['12', '13', '14']

    # 统一变换: resize 到 128x128 (与现有训练管线一致)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道灰度图像
    ])

    print("加载 BITE 训练集...")
    train_dataset = BITEDataset(
        root_dir=root_dir,
        patient_ids=train_patients,
        transform=transform,
    )

    print("加载 BITE 验证集...")
    val_dataset = BITEDataset(
        root_dir=root_dir,
        patient_ids=val_patients,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """测试数据加载器"""
    import matplotlib.pyplot as plt

    root_dir = "./group2 - png"
    train_loader, val_loader = get_bite_data_loaders(root_dir, batch_size=4)

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    # 可视化一个批次
    batch = next(iter(train_loader))
    print(f"\n批次信息:")
    print(f"  Fixed (MR) shape: {batch['fixed'].shape}")
    print(f"  Moving (US) shape: {batch['moving'].shape}")
    print(f"  Patient IDs: {batch['patient_id']}")
    print(f"  Slice indices: {batch['slice_idx']}")

    # 绘图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(min(4, batch['fixed'].shape[0])):
        # 反归一化用于显示
        mr_img = batch['fixed'][i, 0].numpy() * 0.5 + 0.5
        us_img = batch['moving'][i, 0].numpy() * 0.5 + 0.5

        axes[0, i].imshow(mr_img, cmap='gray')
        axes[0, i].set_title(f"MR (Fixed) - P{batch['patient_id'][i]} S{batch['slice_idx'][i]}")
        axes[0, i].axis('off')

        axes[1, i].imshow(us_img, cmap='gray')
        axes[1, i].set_title(f"US (Moving) - P{batch['patient_id'][i]} S{batch['slice_idx'][i]}")
        axes[1, i].axis('off')

    plt.suptitle("BITE Dataset Samples (MR=Fixed, US=Moving)", fontsize=14)
    plt.tight_layout()
    plt.savefig("./figures/bite_dataset_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("样本图像已保存到 ./figures/bite_dataset_samples.png")
