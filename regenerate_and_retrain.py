"""
重新生成形变数据（更大形变）→ 重新训练模型 → 增强可视化
一键运行脚本
"""
import os
import sys
import shutil
import numpy as np
import cv2
import random
from scipy.ndimage import map_coordinates, gaussian_filter
from tqdm import tqdm


# =================== 第一步：重新生成形变数据 ===================

def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


def affine_transform(image, max_rotation=10, max_translation=20, max_scale=0.2):
    rows, cols = image.shape
    angle = random.uniform(-max_rotation, max_rotation)
    tx = random.uniform(-max_translation, max_translation)
    ty = random.uniform(-max_translation, max_translation)
    scale = random.uniform(1 - max_scale, 1 + max_scale)
    center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)


def generate_deformed(input_folder, output_folder, num_per_image,
                      elastic_alpha, elastic_sigma,
                      rotation, translation, scale_range):
    """生成形变图像"""
    os.makedirs(output_folder, exist_ok=True)
    
    # 清空旧文件
    for f in os.listdir(output_folder):
        fp = os.path.join(output_folder, f)
        if os.path.isfile(fp):
            os.remove(fp)
    
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    print(f"从 {input_folder} 读取 {len(image_files)} 张图像，每张生成 {num_per_image} 张形变")
    
    for img_file in tqdm(image_files, desc="生成形变图像"):
        img = cv2.imread(os.path.join(input_folder, img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        base_name = os.path.splitext(img_file)[0]
        
        for i in range(num_per_image):
            deformed = img.copy()
            
            # 仿射变换 (90%概率)
            if random.random() > 0.1:
                rot = random.uniform(rotation[0], rotation[1])
                trans = random.uniform(translation[0], translation[1])
                sc = random.uniform(scale_range[0], scale_range[1])
                deformed = affine_transform(deformed, rot, trans, sc)
            
            # 弹性形变 (90%概率)
            if random.random() > 0.1:
                alpha = random.uniform(elastic_alpha[0], elastic_alpha[1])
                sigma = random.uniform(elastic_sigma[0], elastic_sigma[1])
                deformed = elastic_transform(deformed, alpha, sigma)
            
            out_path = os.path.join(output_folder, f"{base_name}_deformed_{i+1}.png")
            cv2.imwrite(out_path, deformed)
    
    count = len([f for f in os.listdir(output_folder) if f.endswith('.png')])
    print(f"  生成完成: {count} 张形变图像 → {output_folder}")


def step1_regenerate_data():
    """用更大形变参数重新生成数据"""
    print("=" * 60)
    print("第1步: 重新生成形变数据 (更大形变)")
    print("=" * 60)
    
    # 形变参数 — 中等偏大（解剖结构可辨识 + 视觉差异明显）
    params = {
        'elastic_alpha': (60, 150),     # 弹性形变强度
        'elastic_sigma': (10, 15),      # 弹性平滑度 (较平滑，避免撕裂)
        'rotation': (5, 12),            # 旋转角度
        'translation': (5, 15),         # 平移
        'scale_range': (0.04, 0.10),    # 缩放
    }
    
    # 训练集: 每张MRI生成5张形变
    generate_deformed(
        input_folder='./0426_data/train/MRI',
        output_folder='./0426_data/train/MRI_deformed',
        num_per_image=5,
        **params
    )
    
    # 测试集: 每张MRI生成2张形变
    generate_deformed(
        input_folder='./0426_data/test/MRI',
        output_folder='./0426_data/test/MRI_deformed',
        num_per_image=2,
        **params
    )


# =================== 第二步：重新训练 ===================

def step2_retrain():
    """重新训练模型"""
    print("\n" + "=" * 60)
    print("第2步: 重新训练模型")
    print("=" * 60)
    
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import time
    
    from new_model import MultiResolutionRegNet
    from data_loader import MedicalImageDataset, get_data_loaders
    from losses import DualSimilarityLoss
    from utils import save_checkpoint, compute_dice_score
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = get_data_loaders(
        root_dir='./0426_data',
        batch_size=8,
        num_workers=0
    )
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    # 创建模型
    model = MultiResolutionRegNet(in_channels=2).to(device)
    
    # 损失函数
    criterion = DualSimilarityLoss(alpha=10.0, beta=0.5).to(device)
    
    # 优化器 - 使用更高学习率
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # 混合精度
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    save_dir = './0426_new_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    num_epochs = 50  # 增加epoch数
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        for batch in train_loader:
            ct = batch['ct'].to(device)
            mri = batch['mri'].to(device)
            deformed = batch['deformed_mri'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(ct, deformed)
                    loss_dict = criterion(ct, outputs['warped_lvl0'], outputs['flow_lvl0'])
                    # 添加监督损失: warped MRI 应该接近原始 MRI
                    supervised_loss = F.l1_loss(outputs['warped_lvl0'], mri)
                    loss = loss_dict['total'] + 5.0 * supervised_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(ct, deformed)
                loss_dict = criterion(ct, outputs['warped_lvl0'], outputs['flow_lvl0'])
                # 添加监督损失: warped MRI 应该接近原始 MRI
                supervised_loss = F.l1_loss(outputs['warped_lvl0'], mri)
                loss = loss_dict['total'] + 5.0 * supervised_loss
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for batch in val_loader:
                ct = batch['ct'].to(device)
                mri = batch['mri'].to(device)
                deformed = batch['deformed_mri'].to(device)
                outputs = model(ct, deformed)
                loss_dict = criterion(ct, outputs['warped_lvl0'], outputs['flow_lvl0'])
                val_loss += loss_dict['total'].item()
                val_dice += compute_dice_score(outputs['warped_lvl0'], mri)
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        elapsed = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
              f"DICE: {val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"{elapsed:.1f}s")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss
            }, is_best=True, save_path=save_dir)
            print(f"  ★ 保存最佳模型 (val_loss={best_loss:.4f})")
    
    print(f"\n训练完成! 最佳验证损失: {best_loss:.4f}")


# =================== 第三步：增强可视化 ===================

def step3_visualize():
    """运行增强可视化"""
    print("\n" + "=" * 60)
    print("第3步: 生成增强可视化")
    print("=" * 60)
    
    # 直接调用之前的脚本
    import enhanced_registration_viz
    enhanced_registration_viz.main()


# =================== 主函数 ===================

if __name__ == '__main__':
    # 可以单独运行某一步
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == '1':
            step1_regenerate_data()
        elif step == '2':
            step2_retrain()
        elif step == '3':
            step3_visualize()
        else:
            print("用法: python regenerate_and_retrain.py [1|2|3]")
    else:
        # 全部运行
        step1_regenerate_data()
        step2_retrain()
        step3_visualize()
