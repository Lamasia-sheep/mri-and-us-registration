"""
BITE 数据集图像配对 + 训练脚本
================================
步骤:
  1. 图像配对: 对14个病人的 MR-US 切片进行质量筛选和配对
  2. 可视化: 展示配对结果供确认
  3. 训练: 以 MR 为 fixed, US 为 moving 进行配准网络训练

配准方向: MR (fixed) ← US (moving)
"""

import os
import gc
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')          # 非交互式后端，节省内存
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import time

from new_model import MultiResolutionRegNet
from losses import DualSimilarityLoss
from utils import save_image, save_checkpoint


# ====================== 第一步: 图像配对 ======================

def compute_content_ratio(img_path, threshold=15):
    """计算图像的有效内容比例"""
    img = np.array(Image.open(img_path).convert('L'))
    return (img > threshold).sum() / img.size


def compute_mutual_info(mr_img, us_img, bins=32):
    """计算两张图像的互信息 (用于评估配对质量)"""
    hist_2d, _, _ = np.histogram2d(
        mr_img.flatten(), us_img.flatten(), bins=bins
    )
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nonzero = pxy > 0
    mi = (pxy[nonzero] * np.log(pxy[nonzero] / (px_py[nonzero] + 1e-10) + 1e-10)).sum()
    return mi


def pair_bite_dataset(bite_root, content_threshold=0.10, mi_threshold=0.0,
                      intensity_threshold=15):
    """
    对 BITE 数据集进行图像配对和质量筛选。

    筛选标准:
      1. MR 和 US 切片索引相同 (Talairach 空间对齐)
      2. MR 有效内容 > content_threshold
      3. US 有效内容 > content_threshold
      4. 互信息 > mi_threshold (过滤完全无关的配对)

    参数:
        bite_root: BITE 数据集根目录
        content_threshold: 有效内容比例阈值
        mi_threshold: 互信息阈值
        intensity_threshold: 背景像素强度阈值

    返回:
        pairs_info: 配对信息列表
        stats: 统计信息
    """
    print("=" * 60)
    print("BITE 数据集图像配对")
    print("=" * 60)
    print(f"  内容阈值: {content_threshold*100:.0f}%")
    print(f"  互信息阈值: {mi_threshold}")
    print(f"  强度阈值: {intensity_threshold}")
    print()

    all_patients = sorted([
        d for d in os.listdir(bite_root)
        if os.path.isdir(os.path.join(bite_root, d))
    ])

    all_pairs = []
    stats = defaultdict(int)
    patient_stats = {}

    for pid in all_patients:
        pdir = os.path.join(bite_root, pid)
        subdirs = sorted(os.listdir(pdir))
        mr_dirs = [s for s in subdirs if 'mr' in s.lower() and os.path.isdir(os.path.join(pdir, s))]
        us_dirs = [s for s in subdirs if 'us' in s.lower() and os.path.isdir(os.path.join(pdir, s))]

        if not mr_dirs or not us_dirs:
            print(f"  [跳过] 病人 {pid}: 缺少 MR 或 US 目录")
            continue

        mr_dir = os.path.join(pdir, mr_dirs[0])
        us_dir = os.path.join(pdir, us_dirs[0])

        # 获取所有切片
        mr_files = {int(f.replace('.png', '')): os.path.join(mr_dir, f)
                    for f in os.listdir(mr_dir) if f.endswith('.png')}
        us_files = {int(f.replace('.png', '')): os.path.join(us_dir, f)
                    for f in os.listdir(us_dir) if f.endswith('.png')}

        common_indices = sorted(set(mr_files.keys()) & set(us_files.keys()))
        stats['total_common'] += len(common_indices)

        patient_pairs = []
        rejected_content = 0
        rejected_mi = 0

        for idx in common_indices:
            mr_path = mr_files[idx]
            us_path = us_files[idx]

            # 检查内容比例
            mr_ratio = compute_content_ratio(mr_path, intensity_threshold)
            us_ratio = compute_content_ratio(us_path, intensity_threshold)

            if mr_ratio < content_threshold or us_ratio < content_threshold:
                rejected_content += 1
                continue

            # 计算互信息
            mr_img = np.array(Image.open(mr_path).convert('L'))
            us_img = np.array(Image.open(us_path).convert('L'))
            # resize to same size for MI computation
            from PIL import Image as PILImage
            us_resized = np.array(PILImage.open(us_path).convert('L').resize(
                (mr_img.shape[1], mr_img.shape[0]), PILImage.BILINEAR
            ))
            mi = compute_mutual_info(mr_img, us_resized)

            if mi < mi_threshold:
                rejected_mi += 1
                continue

            pair_info = {
                'patient_id': pid,
                'slice_idx': idx,
                'mr_path': mr_path,
                'us_path': us_path,
                'mr_content': float(mr_ratio),
                'us_content': float(us_ratio),
                'mutual_info': float(mi)
            }
            patient_pairs.append(pair_info)

        all_pairs.extend(patient_pairs)
        patient_stats[pid] = {
            'total': len(common_indices),
            'valid': len(patient_pairs),
            'rejected_content': rejected_content,
            'rejected_mi': rejected_mi,
            'avg_mi': np.mean([p['mutual_info'] for p in patient_pairs]) if patient_pairs else 0
        }

        print(f"  病人 {pid}: {len(patient_pairs)}/{len(common_indices)} 有效配对 "
              f"(弃 content={rejected_content}, mi={rejected_mi}), "
              f"avg MI={patient_stats[pid]['avg_mi']:.4f}")

    stats['total_pairs'] = len(all_pairs)
    stats['num_patients'] = len(patient_stats)

    print(f"\n{'='*60}")
    print(f"配对完成: {len(all_pairs)} 个有效 MR-US 配对 (来自 {len(patient_stats)} 个病人)")
    print(f"{'='*60}\n")

    return all_pairs, patient_stats


def visualize_pairs(pairs, save_path, num_samples=14):
    """美化的配对样本可视化"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 每个病人选中间切片
    patients = sorted(set(p['patient_id'] for p in pairs))
    selected = []
    for pid in patients:
        p_pairs = [p for p in pairs if p['patient_id'] == pid]
        mid = len(p_pairs) // 2
        selected.append(p_pairs[mid])
        if len(selected) >= num_samples:
            break

    n = len(selected)
    cols = min(7, n)
    rows_groups = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_groups * 2, cols,
                              figsize=(3.2 * cols, 3.5 * rows_groups * 2))
    if rows_groups * 2 == 2:
        axes = axes.reshape(2, cols)

    for i, pair in enumerate(selected):
        row_base = (i // cols) * 2
        col = i % cols

        mr_img = np.array(Image.open(pair['mr_path']).convert('L')).astype(np.float32) / 255.0
        us_img = np.array(Image.open(pair['us_path']).convert('L')).astype(np.float32) / 255.0

        # MR
        axes[row_base, col].imshow(mr_img, cmap='gray', vmin=0, vmax=1)
        axes[row_base, col].set_title(
            f"MR  P{pair['patient_id']}  S{pair['slice_idx']}",
            fontsize=8, fontweight='bold', color='#1565C0'
        )
        # 在图像左下角标注内容占比
        axes[row_base, col].text(
            2, mr_img.shape[0]-4, f"content {pair['mr_content']:.0%}",
            fontsize=6, color='lime', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6)
        )
        axes[row_base, col].axis('off')

        # US
        axes[row_base+1, col].imshow(us_img, cmap='gray', vmin=0, vmax=1)
        axes[row_base+1, col].set_title(
            f"US  MI={pair['mutual_info']:.3f}",
            fontsize=8, fontweight='bold', color='#E65100'
        )
        axes[row_base+1, col].text(
            2, us_img.shape[0]-4, f"content {pair['us_content']:.0%}",
            fontsize=6, color='cyan', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6)
        )
        axes[row_base+1, col].axis('off')

    # 关闭多余的axes
    total_slots = rows_groups * cols
    for i in range(n, total_slots):
        row_base = (i // cols) * 2
        col = i % cols
        if row_base < rows_groups * 2 and col < cols:
            axes[row_base, col].axis('off')
            axes[row_base+1, col].axis('off')

    fig.suptitle(
        f"BITE Dataset Pairing Results  |  {n} patients  |  MR (Fixed) ↔ US (Moving)",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"配对可视化已保存: {save_path}")


# ====================== 第二步: 数据集 ======================

class BITEPairedDataset(Dataset):
    """基于预配对结果的 BITE 数据集"""

    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        mr_image = Image.open(pair['mr_path']).convert('L')
        us_image = Image.open(pair['us_path']).convert('L')

        if self.transform:
            mr_image = self.transform(mr_image)
            us_image = self.transform(us_image)

        return {
            'fixed': mr_image,      # MRI 固定图像
            'moving': us_image,     # US 移动图像
            'patient_id': pair['patient_id'],
            'slice_idx': pair['slice_idx'],
            'mutual_info': pair['mutual_info']
        }


# ====================== 第三步: 训练 ======================

def compute_ssim(x, y, window_size=11):
    """计算 SSIM"""
    x_gray, y_gray = x, y
    if x.shape[2] > 64:
        x_gray = F.avg_pool2d(x_gray, kernel_size=2, stride=2)
        y_gray = F.avg_pool2d(y_gray, kernel_size=2, stride=2)

    def _gw(ws, sigma=1.5):
        g = torch.exp(-torch.arange(ws).float().div(ws//2).pow(2).mul(2.0).div(2*sigma*sigma))
        return g / g.sum()

    gk = _gw(window_size)
    w = gk.unsqueeze(1) * gk.unsqueeze(0)
    w = w.unsqueeze(0).unsqueeze(0).to(x.device).expand(1, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(F.pad(x_gray, [pad]*4, mode='replicate'), w, groups=1)
    mu2 = F.conv2d(F.pad(y_gray, [pad]*4, mode='replicate'), w, groups=1)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(F.pad(x_gray*x_gray, [pad]*4, mode='replicate'), w, groups=1) - mu1_sq
    s2 = F.conv2d(F.pad(y_gray*y_gray, [pad]*4, mode='replicate'), w, groups=1) - mu2_sq
    s12 = F.conv2d(F.pad(x_gray*y_gray, [pad]*4, mode='replicate'), w, groups=1) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2+C1)*(2*s12+C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return ssim_map.mean().item()


def compute_nmi(x, y, bins=64):
    """计算 NMI"""
    xn = x.detach().cpu().numpy().flatten()
    yn = y.detach().cpu().numpy().flatten()
    h2d, _, _ = np.histogram2d(xn, yn, bins=bins)
    h2d = h2d / h2d.sum()
    px, py = h2d.sum(1), h2d.sum(0)
    px_py = px[:, None] * py[None, :]
    nz = h2d > 0
    mi = (h2d[nz] * np.log(h2d[nz] / px_py[nz])).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return 2 * mi / (hx + hy + 1e-10)


def make_checkerboard_overlay(img_a, img_b, block_size=16):
    """创建棋盘格叠加图，用于直观对比两张图像的对齐程度"""
    H, W = img_a.shape[:2]
    board = np.zeros((H, W), dtype=bool)
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                board[i:i+block_size, j:j+block_size] = True
    result = np.where(board, img_a, img_b)
    return result


def make_color_overlay(fixed_np, warped_np):
    """创建红绿叠加图：fixed=绿，warped=红，重合=黄"""
    H, W = fixed_np.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[..., 0] = warped_np   # R = warped US
    overlay[..., 1] = fixed_np    # G = fixed MR
    overlay[..., 2] = 0
    return np.clip(overlay, 0, 1)


def flow_to_rgb(flow_np):
    """将 (2,H,W) 变形场转为 HSV 彩色图"""
    import cv2
    u, v = flow_np[0], flow_np[1]
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    max_mag = mag.max() if mag.max() > 0 else 1.0
    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb.astype(np.float32) / 255.0


def draw_deformation_grid(flow_np, grid_step=8):
    """在白色背景上画变形网格"""
    u, v = flow_np[0], flow_np[1]
    H, W = u.shape
    canvas = np.ones((H, W, 3), dtype=np.float32)
    # 水平线
    for i in range(0, H, grid_step):
        for j in range(W - 1):
            y1 = int(np.clip(i + v[i, j], 0, H-1))
            x1 = int(np.clip(j + u[i, j], 0, W-1))
            y2 = int(np.clip(i + v[i, j+1], 0, H-1))
            x2 = int(np.clip(j+1 + u[i, j+1], 0, W-1))
            import cv2
            cv2.line(canvas, (x1, y1), (x2, y2), (0.2, 0.4, 0.8), 1, cv2.LINE_AA)
    # 垂直线
    for j in range(0, W, grid_step):
        for i in range(H - 1):
            y1 = int(np.clip(i + v[i, j], 0, H-1))
            x1 = int(np.clip(j + u[i, j], 0, W-1))
            y2 = int(np.clip(i+1 + v[i+1, j], 0, H-1))
            x2 = int(np.clip(j + u[i+1, j], 0, W-1))
            import cv2
            cv2.line(canvas, (x1, y1), (x2, y2), (0.2, 0.4, 0.8), 1, cv2.LINE_AA)
    return canvas


def save_beautiful_result(fixed, moving, warped, flow, pid, sid,
                          nmi_before, nmi_after, ssim_before, ssim_after,
                          save_path, epoch=None):
    """
    生成美化的配准结果可视化图（论文级质量）

    布局 (2行 × 4列):
      Row 1: MR(Fixed) | US(Moving) | Warped US | Deformation Field
      Row 2: Checkerboard(MR/US_before) | Checkerboard(MR/Warped) | Color Overlay | Deformation Grid
    """
    import matplotlib
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['axes.titleweight'] = 'bold'

    # 反归一化 → [0,1]
    def to_np(t):
        x = t.detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        mn, mx = x.min(), x.max()
        if mx > mn:
            x = (x - mn) / (mx - mn)
        return x

    f_np = to_np(fixed)
    m_np = to_np(moving)
    w_np = to_np(warped)
    flow_np = flow.detach().cpu().numpy()
    if flow_np.ndim == 4:
        flow_np = flow_np[0]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9),
                              gridspec_kw={'hspace': 0.28, 'wspace': 0.12})

    # ---------- Row 1 ----------
    # (0,0) MR Fixed
    axes[0,0].imshow(f_np, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title('MR (Fixed)', color='#2196F3')

    # (0,1) US Moving
    axes[0,1].imshow(m_np, cmap='gray', vmin=0, vmax=1)
    axes[0,1].set_title('US (Moving)', color='#FF9800')

    # (0,2) Warped US
    axes[0,2].imshow(w_np, cmap='gray', vmin=0, vmax=1)
    axes[0,2].set_title('Warped US', color='#4CAF50')

    # (0,3) Deformation field (HSV color-coded)
    flow_rgb = flow_to_rgb(flow_np)
    axes[0,3].imshow(flow_rgb)
    axes[0,3].set_title('Deformation Field', color='#9C27B0')

    # ---------- Row 2 ----------
    # (1,0) Checkerboard before
    cb_before = make_checkerboard_overlay(f_np, m_np, block_size=16)
    axes[1,0].imshow(cb_before, cmap='gray', vmin=0, vmax=1)
    axes[1,0].set_title('Checker: MR / US (Before)')

    # (1,1) Checkerboard after
    cb_after = make_checkerboard_overlay(f_np, w_np, block_size=16)
    axes[1,1].imshow(cb_after, cmap='gray', vmin=0, vmax=1)
    axes[1,1].set_title('Checker: MR / Warped (After)')

    # (1,2) Color overlay (R=warped, G=fixed)
    overlay = make_color_overlay(f_np, w_np)
    axes[1,2].imshow(overlay)
    axes[1,2].set_title('Overlay (G=MR, R=Warped)')

    # (1,3) Deformation grid
    grid_img = draw_deformation_grid(flow_np, grid_step=6)
    axes[1,3].imshow(grid_img)
    axes[1,3].set_title('Deformation Grid')

    # 去掉所有坐标轴
    for ax in axes.flat:
        ax.axis('off')

    # 顶部标题
    ep_str = f'  |  Epoch {epoch}' if epoch is not None else ''
    fig.suptitle(
        f'Patient {pid}  Slice {sid}{ep_str}\n'
        f'NMI: {nmi_before:.4f} → {nmi_after:.4f} '
        f'({"+" if nmi_after>=nmi_before else ""}{nmi_after-nmi_before:.4f})   |   '
        f'SSIM: {ssim_before:.4f} → {ssim_after:.4f} '
        f'({"+" if ssim_after>=ssim_before else ""}{ssim_after-ssim_before:.4f})',
        fontsize=13, fontweight='bold', y=0.99
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


def validate(model, val_loader, criterion, device, epoch, save_dir=None, max_vis=8):
    """验证 (带美化可视化)"""
    model.eval()
    losses, nmi_b, nmi_a, ssim_b, ssim_a = [], [], [], [], []
    vis_count = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"验证 Ep{epoch+1}", leave=False):
            fixed = batch['fixed'].to(device)
            moving = batch['moving'].to(device)

            outputs = model(fixed, moving)
            warped = outputs['warped_lvl0']
            flow = outputs['flow_lvl0']

            loss_dict = criterion(fixed, warped, flow)
            losses.append(loss_dict['total'].item())

            for b in range(fixed.shape[0]):
                _nmi_b = compute_nmi(fixed[b], moving[b])
                _nmi_a = compute_nmi(fixed[b], warped[b])
                _ssim_b = compute_ssim(fixed[b:b+1], moving[b:b+1])
                _ssim_a = compute_ssim(fixed[b:b+1], warped[b:b+1])
                nmi_b.append(_nmi_b); nmi_a.append(_nmi_a)
                ssim_b.append(_ssim_b); ssim_a.append(_ssim_a)

                # 美化可视化
                if save_dir and vis_count < max_vis:
                    pid = batch['patient_id'][b]
                    sid = batch['slice_idx'][b].item() if torch.is_tensor(batch['slice_idx'][b]) else batch['slice_idx'][b]
                    save_beautiful_result(
                        fixed[b], moving[b], warped[b], flow[b:b+1],
                        pid, sid, _nmi_b, _nmi_a, _ssim_b, _ssim_a,
                        save_path=os.path.join(save_dir, f"val_P{pid}_S{sid}.png"),
                        epoch=epoch+1
                    )
                    vis_count += 1

    results = {
        'loss': np.mean(losses),
        'nmi_before': np.mean(nmi_b), 'nmi_after': np.mean(nmi_a),
        'ssim_before': np.mean(ssim_b), 'ssim_after': np.mean(ssim_a),
        'nmi_improve': np.mean(nmi_a) - np.mean(nmi_b),
        'ssim_improve': np.mean(ssim_a) - np.mean(ssim_b),
        'num_samples': len(nmi_b)
    }

    print(f"  验证: loss={results['loss']:.4f}, "
          f"NMI {results['nmi_before']:.4f}→{results['nmi_after']:.4f} "
          f"({'↑' if results['nmi_improve']>0 else '↓'}{abs(results['nmi_improve']):.4f}), "
          f"SSIM {results['ssim_before']:.4f}→{results['ssim_after']:.4f} "
          f"({'↑' if results['ssim_improve']>0 else '↓'}{abs(results['ssim_improve']):.4f})")

    return results


def train(model, train_loader, val_loader, criterion, optimizer, device, config):
    """训练主循环"""
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    best_loss = float('inf')
    history = []

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        epoch_start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in progress:
            fixed = batch['fixed'].to(device, non_blocking=True)
            moving = batch['moving'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if config.get('use_amp', False) and torch.cuda.is_available():
                from torch.cuda.amp import autocast, GradScaler
                with autocast():
                    outputs = model(fixed, moving)
                    loss_dict = criterion(fixed, outputs['warped_lvl0'], outputs['flow_lvl0'])
                    loss = loss_dict['total']
                config['scaler'].scale(loss).backward()
                config['scaler'].step(optimizer)
                config['scaler'].update()
            else:
                outputs = model(fixed, moving)
                loss_dict = criterion(fixed, outputs['warped_lvl0'], outputs['flow_lvl0'])
                loss = loss_dict['total']
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mi': f"{loss_dict['mi'].item():.4f}",
                'mind': f"{loss_dict['mind'].item():.4f}",
                'reg': f"{loss_dict['reg'].item():.4f}"
            })

        avg_train_loss = epoch_loss / batch_count
        scheduler.step()
        epoch_time = time.time() - epoch_start

        # 每个 epoch 结束后清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 验证
        val_freq = config.get('val_frequency', 5)
        if (epoch + 1) % val_freq == 0 or epoch == config['num_epochs'] - 1:
            val_results = validate(
                model, val_loader, criterion, device, epoch,
                save_dir=os.path.join(config['result_dir'], f"val_ep{epoch+1}")
            )

            history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_results['loss'],
                'nmi_after': val_results['nmi_after'],
                'ssim_after': val_results['ssim_after'],
                'nmi_improve': val_results['nmi_improve'],
                'ssim_improve': val_results['ssim_improve']
            })

            # 保存最佳模型
            if val_results['loss'] < best_loss:
                best_loss = val_results['loss']
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'val_results': val_results
                }, is_best=True, save_path=config['checkpoint_dir'])
                print(f"  ✓ 保存最佳模型 (loss={best_loss:.4f})")

            print(f"  Epoch {epoch+1}: train={avg_train_loss:.4f}, val={val_results['loss']:.4f}, "
                  f"time={epoch_time:.1f}s, lr={scheduler.get_last_lr()[0]:.6f}")
        else:
            print(f"  Epoch {epoch+1}: train={avg_train_loss:.4f}, time={epoch_time:.1f}s")

    # 保存训练历史
    if history:
        with open(os.path.join(config['result_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        plot_training_history(history, config['result_dir'])

    return best_loss, history


def plot_training_history(history, save_dir):
    """绘制美化的训练曲线"""
    import matplotlib
    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
    })

    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 颜色方案
    c_train = '#1976D2'
    c_val   = '#D32F2F'
    c_nmi   = '#388E3C'
    c_ssim  = '#7B1FA2'
    c_imp   = '#FF6F00'

    # ---- Loss ----
    ax = axes[0]
    ax.plot(epochs, [h['train_loss'] for h in history],
            '-o', color=c_train, label='Train Loss', markersize=5, linewidth=2)
    ax.plot(epochs, [h['val_loss'] for h in history],
            '-s', color=c_val, label='Val Loss', markersize=5, linewidth=2)
    ax.fill_between(epochs,
                    [h['train_loss'] for h in history],
                    [h['val_loss'] for h in history],
                    alpha=0.1, color='gray')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- NMI ----
    ax = axes[1]
    ax.plot(epochs, [h['nmi_after'] for h in history],
            '-o', color=c_nmi, label='NMI (After Reg.)', markersize=5, linewidth=2)
    ax2 = ax.twinx()
    ax2.bar(epochs, [h['nmi_improve'] for h in history],
            width=0.6, alpha=0.35, color=c_imp, label='NMI Δ')
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NMI', color=c_nmi)
    ax2.set_ylabel('NMI Improvement', color=c_imp)
    ax.set_title('Normalized Mutual Information')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='lower right',
              framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)

    # ---- SSIM ----
    ax = axes[2]
    ax.plot(epochs, [h['ssim_after'] for h in history],
            '-o', color=c_ssim, label='SSIM (After Reg.)', markersize=5, linewidth=2)
    ax2 = ax.twinx()
    ax2.bar(epochs, [h['ssim_improve'] for h in history],
            width=0.6, alpha=0.35, color=c_imp, label='SSIM Δ')
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM', color=c_ssim)
    ax2.set_ylabel('SSIM Improvement', color=c_imp)
    ax.set_title('Structural Similarity')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='lower right',
              framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)

    fig.suptitle('BITE Dataset Training Progress (MR=Fixed, US=Moving)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"训练曲线已保存: {os.path.join(save_dir, 'training_curves.png')}")


# ====================== 主程序 ======================

if __name__ == "__main__":
    BITE_ROOT = "./group2 - png"
    OUTPUT_DIR = "./bite_train_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # ============ 第一步: 图像配对 ============
    print("【第一步】图像配对与质量筛选\n")
    all_pairs, patient_stats = pair_bite_dataset(
        BITE_ROOT,
        content_threshold=0.10,   # 至少 10% 有效内容
        mi_threshold=0.0,         # 互信息 > 0
        intensity_threshold=15    # 像素值 > 15 才算有效
    )

    # 按病人划分 train/val: 前11个训练, 后3个验证
    train_patients = {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'}
    val_patients = {'12', '13', '14'}

    train_pairs = [p for p in all_pairs if p['patient_id'] in train_patients]
    val_pairs = [p for p in all_pairs if p['patient_id'] in val_patients]

    print(f"\n数据划分:")
    print(f"  训练集: {len(train_pairs)} 对 (病人 01-11)")
    print(f"  验证集: {len(val_pairs)} 对 (病人 12-14)")

    # 保存配对信息
    pair_info_path = os.path.join(OUTPUT_DIR, 'pair_info.json')
    with open(pair_info_path, 'w') as f:
        json.dump({
            'train_count': len(train_pairs),
            'val_count': len(val_pairs),
            'patient_stats': patient_stats,
            'train_pairs': train_pairs,
            'val_pairs': val_pairs
        }, f, indent=2, ensure_ascii=False)
    print(f"配对信息已保存: {pair_info_path}")

    # ============ 第二步: 可视化配对 ============
    print("\n【第二步】可视化配对结果\n")
    visualize_pairs(
        all_pairs,
        save_path=os.path.join(OUTPUT_DIR, 'pair_visualization.png'),
        num_samples=14  # 每个病人一个
    )

    # ============ 第三步: 训练 ============
    print("\n【第三步】训练配准网络 (MR=fixed, US=moving)\n")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = BITEPairedDataset(train_pairs, transform=transform)
    val_dataset = BITEPairedDataset(val_pairs, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"训练: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证: {len(val_dataset)} 样本, {len(val_loader)} 批次\n")

    # 创建模型
    model = MultiResolutionRegNet(in_channels=2).to(device)
    criterion = DualSimilarityLoss(alpha=10.0, beta=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})\n")

    config = {
        'num_epochs': 30,
        'val_frequency': 3,       # 每3个epoch验证一次，产生更多可视化
        'checkpoint_dir': os.path.join(OUTPUT_DIR, 'checkpoints'),
        'result_dir': os.path.join(OUTPUT_DIR, 'results'),
    }
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)

    print(f"训练配置:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  学习率: 1e-4 (Cosine退火)")
    print(f"  损失: NMI + MIND + Smooth (α=10.0, β=0.5)")
    print(f"  验证频率: 每 {config['val_frequency']} epoch")
    print(f"  输出目录: {OUTPUT_DIR}")
    print()

    # 开始训练
    start_time = time.time()
    best_loss, history = train(model, train_loader, val_loader, criterion,
                               optimizer, device, config)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  最佳验证损失: {best_loss:.4f}")
    if history:
        best_h = min(history, key=lambda h: h['val_loss'])
        print(f"  最佳 Epoch: {best_h['epoch']}")
        print(f"  最佳 NMI: {best_h['nmi_after']:.4f} (↑{best_h['nmi_improve']:.4f})")
        print(f"  最佳 SSIM: {best_h['ssim_after']:.4f} (↑{best_h['ssim_improve']:.4f})")
    print(f"{'='*60}")
