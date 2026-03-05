"""
RESECT + BITE 联合训练脚本
===========================
合并两个数据集训练 MR-US 配准网络 (SelectiveSSM-LTMNet)

数据来源:
  - RESECT (23 例): resect_png/CaseXX/MRI_T1/ + US_before/
  - BITE   (14 例): group2 - png/XX/XXX_mr_tal_png/ + XXXa_us_tal_png/

配准方向: MR (fixed) ← US (moving)
模型:     SelectiveSSM-LTMNet (new_model.py)
"""

import os
import gc
import json
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import time

from new_model import MultiResolutionRegNet
from losses import DualSimilarityLoss
from utils import save_checkpoint

# ==================== 工具函数 ====================

def compute_content_ratio(img_path, threshold=15):
    """计算图像有效内容比例"""
    img = np.array(Image.open(img_path).convert('L'))
    return (img > threshold).sum() / img.size


def compute_mutual_info(img1, img2, bins=32, resize_to=(128, 128)):
    """计算两张图像的互信息（先统一尺寸，避免原始分辨率不一致）"""
    if resize_to is not None:
        if img1.shape != resize_to:
            img1 = np.array(Image.fromarray(img1.astype(np.uint8)).resize(resize_to, Image.BILINEAR))
        if img2.shape != resize_to:
            img2 = np.array(Image.fromarray(img2.astype(np.uint8)).resize(resize_to, Image.BILINEAR))
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)
    pxy = hist_2d / (hist_2d.sum() + 1e-10)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nonzero = pxy > 0
    mi = (pxy[nonzero] * np.log(pxy[nonzero] / (px_py[nonzero] + 1e-10) + 1e-10)).sum()
    return mi


# ==================== RESECT 配对 ====================

def pair_resect_dataset(resect_root, content_threshold=0.10,
                        max_pairs_per_case=60, local_search_window=6, mi_threshold=0.02):
    """
    RESECT 数据集配对:
    - MRI_T1 为 fixed, US_before 为 moving
    - 由于 MRI 和 US 来自不同体数据 (切片索引无对应关系),
      采用按比例位置匹配 + 质量筛选策略
    """
    print("=" * 60)
    print("RESECT 数据集配对 (MRI_T1 ↔ US_before)")
    print("=" * 60)

    all_pairs = []

    cases = sorted([d for d in os.listdir(resect_root)
                    if os.path.isdir(os.path.join(resect_root, d)) and d.startswith('Case')])

    for case in cases:
        t1_dir = os.path.join(resect_root, case, 'MRI_T1')
        us_dir = os.path.join(resect_root, case, 'US_before')

        if not os.path.isdir(t1_dir) or not os.path.isdir(us_dir):
            print(f"  [跳过] {case}: 缺少 MRI_T1 或 US_before")
            continue

        # 获取所有有效切片 (有足够内容的)
        t1_files = sorted([f for f in os.listdir(t1_dir) if f.endswith('.png')],
                          key=lambda x: int(x.replace('.png', '')))
        us_files = sorted([f for f in os.listdir(us_dir) if f.endswith('.png')],
                          key=lambda x: int(x.replace('.png', '')))

        # 筛选有效 MRI 切片
        valid_t1 = []
        for f in t1_files:
            path = os.path.join(t1_dir, f)
            if compute_content_ratio(path) >= content_threshold:
                valid_t1.append(path)

        # 筛选有效 US 切片
        valid_us = []
        for f in us_files:
            path = os.path.join(us_dir, f)
            if compute_content_ratio(path) >= content_threshold:
                valid_us.append(path)

        if not valid_t1 or not valid_us:
            print(f"  [跳过] {case}: 有效切片不足 (T1={len(valid_t1)}, US={len(valid_us)})")
            continue

        # 按比例位置粗配 + 局部窗口内 MI 最优细配
        case_pairs = []
        n_t1, n_us = len(valid_t1), len(valid_us)
        n_proportional = min(n_t1, n_us, max_pairs_per_case)

        for k in range(n_proportional):
            t1_idx = int(k * n_t1 / n_proportional)
            us_center = int(k * n_us / n_proportional)
            t1_path = valid_t1[t1_idx]

            # 在局部窗口中选 MI 最高的 US 切片，降低错配概率
            best_us_path = None
            best_mi = -1e9
            left = max(0, us_center - local_search_window)
            right = min(n_us - 1, us_center + local_search_window)

            t1_img = np.array(Image.open(t1_path).convert('L'))
            for us_idx in range(left, right + 1):
                us_path_cand = valid_us[us_idx]
                us_img = np.array(Image.open(us_path_cand).convert('L'))
                mi_val = compute_mutual_info(t1_img, us_img, bins=32, resize_to=(128, 128))
                if mi_val > best_mi:
                    best_mi = mi_val
                    best_us_path = us_path_cand

            if best_us_path is None or best_mi < mi_threshold:
                continue

            t1_slice = int(os.path.basename(t1_path).replace('.png', ''))
            us_slice = int(os.path.basename(best_us_path).replace('.png', ''))

            case_pairs.append({
                'patient_id': case,
                'slice_idx': f"T{t1_slice}_U{us_slice}",
                'mr_path': t1_path,
                'us_path': best_us_path,
                'source': 'RESECT'
            })

        all_pairs.extend(case_pairs)
        print(f"  {case}: {len(case_pairs)} 配对 (T1={n_t1}, US={n_us} 有效切片)")

    print(f"\n  RESECT 总计: {len(all_pairs)} 配对 (来自 {len(cases)} 例)\n")
    return all_pairs


# ==================== BITE 配对 ====================

def pair_bite_dataset(bite_root, content_threshold=0.10, mi_threshold=0.03):
    """
    BITE 数据集配对 (切片索引对齐):
    - MR 为 fixed, US 为 moving
    """
    print("=" * 60)
    print("BITE 数据集配对 (MR ↔ US)")
    print("=" * 60)

    all_pairs = []
    patients = sorted([d for d in os.listdir(bite_root)
                       if os.path.isdir(os.path.join(bite_root, d))])

    for pid in patients:
        pdir = os.path.join(bite_root, pid)
        subdirs = sorted(os.listdir(pdir))
        mr_dirs = [s for s in subdirs if 'mr' in s.lower() and os.path.isdir(os.path.join(pdir, s))]
        us_dirs = [s for s in subdirs if 'us' in s.lower() and os.path.isdir(os.path.join(pdir, s))]

        if not mr_dirs or not us_dirs:
            continue

        mr_dir = os.path.join(pdir, mr_dirs[0])
        us_dir = os.path.join(pdir, us_dirs[0])

        mr_files = {int(f.replace('.png', '')): os.path.join(mr_dir, f)
                    for f in os.listdir(mr_dir) if f.endswith('.png')}
        us_files = {int(f.replace('.png', '')): os.path.join(us_dir, f)
                    for f in os.listdir(us_dir) if f.endswith('.png')}

        common = sorted(set(mr_files) & set(us_files))
        patient_pairs = []

        for idx in common:
            mr_path, us_path = mr_files[idx], us_files[idx]
            mr_ratio = compute_content_ratio(mr_path)
            us_ratio = compute_content_ratio(us_path)

            if mr_ratio < content_threshold or us_ratio < content_threshold:
                continue

            mr_img = np.array(Image.open(mr_path).convert('L'))
            us_img = np.array(Image.open(us_path).convert('L'))
            mi_val = compute_mutual_info(mr_img, us_img, bins=32, resize_to=(128, 128))
            if mi_val < mi_threshold:
                continue

            patient_pairs.append({
                'patient_id': f"BITE_{pid}",
                'slice_idx': idx,
                'mr_path': mr_path,
                'us_path': us_path,
                'source': 'BITE'
            })

        all_pairs.extend(patient_pairs)
        print(f"  病人 {pid}: {len(patient_pairs)}/{len(common)} 有效配对")

    print(f"\n  BITE 总计: {len(all_pairs)} 配对\n")
    return all_pairs


# ==================== 数据集 ====================

class CombinedPairedDataset(Dataset):
    """合并 RESECT + BITE 数据的配对数据集"""

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
            'fixed': mr_image,
            'moving': us_image,
            'patient_id': str(pair['patient_id']),
            'slice_idx': str(pair['slice_idx']),
            'source': pair['source']
        }


# ==================== 评估指标 ====================

def compute_ssim(x, y, window_size=11):
    """计算 SSIM"""
    x_gray, y_gray = x, y
    if x.shape[2] > 64:
        x_gray = F.avg_pool2d(x_gray, 2, 2)
        y_gray = F.avg_pool2d(y_gray, 2, 2)

    def _gw(ws, sigma=1.5):
        g = torch.exp(-torch.arange(ws).float().div(ws//2).pow(2).mul(2.0).div(2*sigma*sigma))
        return g / g.sum()

    gk = _gw(window_size)
    w = (gk.unsqueeze(1) * gk.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(x.device)
    w = w.expand(1, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(F.pad(x_gray, [pad]*4, mode='replicate'), w)
    mu2 = F.conv2d(F.pad(y_gray, [pad]*4, mode='replicate'), w)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(F.pad(x_gray*x_gray, [pad]*4, mode='replicate'), w) - mu1_sq
    s2 = F.conv2d(F.pad(y_gray*y_gray, [pad]*4, mode='replicate'), w) - mu2_sq
    s12 = F.conv2d(F.pad(x_gray*y_gray, [pad]*4, mode='replicate'), w) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2+C1)*(2*s12+C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return ssim_map.mean().item()


def compute_nmi(x, y, bins=64):
    """计算 NMI"""
    xn = x.detach().cpu().numpy().flatten()
    yn = y.detach().cpu().numpy().flatten()
    h2d, _, _ = np.histogram2d(xn, yn, bins=bins)
    h2d = h2d / (h2d.sum() + 1e-10)
    px, py = h2d.sum(1), h2d.sum(0)
    px_py = px[:, None] * py[None, :]
    nz = h2d > 0
    mi = (h2d[nz] * np.log(h2d[nz] / (px_py[nz] + 1e-10))).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return 2 * mi / (hx + hy + 1e-10)


# ==================== 可视化 ====================

def flow_to_rgb(flow_np):
    """(2,H,W) 流场 → HSV 彩色图"""
    import cv2
    u, v = flow_np[0], flow_np[1]
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    max_mag = mag.max() if mag.max() > 0 else 1.0
    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def make_checkerboard(img_a, img_b, block=16):
    """棋盘格叠加"""
    H, W = img_a.shape[:2]
    board = np.zeros((H, W), dtype=bool)
    for i in range(0, H, block):
        for j in range(0, W, block):
            if ((i // block) + (j // block)) % 2 == 0:
                board[i:i+block, j:j+block] = True
    return np.where(board, img_a, img_b)


def make_color_overlay(fixed_np, warped_np):
    """红绿叠加图"""
    H, W = fixed_np.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[..., 0] = warped_np
    overlay[..., 1] = fixed_np
    return np.clip(overlay, 0, 1)


def draw_deformation_grid(flow_np, grid_step=6):
    """变形网格"""
    import cv2
    u, v = flow_np[0], flow_np[1]
    H, W = u.shape
    canvas = np.ones((H, W, 3), dtype=np.float32)
    for i in range(0, H, grid_step):
        for j in range(W - 1):
            y1 = int(np.clip(i + v[i, j], 0, H-1))
            x1 = int(np.clip(j + u[i, j], 0, W-1))
            y2 = int(np.clip(i + v[i, j+1], 0, H-1))
            x2 = int(np.clip(j+1 + u[i, j+1], 0, W-1))
            cv2.line(canvas, (x1, y1), (x2, y2), (0.2, 0.4, 0.8), 1, cv2.LINE_AA)
    for j in range(0, W, grid_step):
        for i in range(H - 1):
            y1 = int(np.clip(i + v[i, j], 0, H-1))
            x1 = int(np.clip(j + u[i, j], 0, W-1))
            y2 = int(np.clip(i+1 + v[i+1, j], 0, H-1))
            x2 = int(np.clip(j + u[i+1, j], 0, W-1))
            cv2.line(canvas, (x1, y1), (x2, y2), (0.2, 0.4, 0.8), 1, cv2.LINE_AA)
    return canvas


def save_beautiful_result(fixed, moving, warped, flow, pid, sid, source,
                          nmi_before, nmi_after, ssim_before, ssim_after,
                          save_path, epoch=None):
    """
    论文级配准结果可视化 (2行 × 4列)
    Row 1: MR(Fixed) | US(Moving) | Warped US | Deformation Field
    Row 2: Checker(Before) | Checker(After) | Color Overlay | Deformation Grid
    """
    matplotlib.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})

    def to_np(t):
        x = t.detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        mn, mx = x.min(), x.max()
        if mx > mn:
            x = (x - mn) / (mx - mn)
        return x

    f_np, m_np, w_np = to_np(fixed), to_np(moving), to_np(warped)
    flow_np = flow.detach().cpu().numpy()
    if flow_np.ndim == 4:
        flow_np = flow_np[0]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9),
                              gridspec_kw={'hspace': 0.28, 'wspace': 0.12})

    # Row 1
    axes[0, 0].imshow(f_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('MR (Fixed)', color='#2196F3')

    axes[0, 1].imshow(m_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('US (Moving)', color='#FF9800')

    axes[0, 2].imshow(w_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Warped US', color='#4CAF50')

    axes[0, 3].imshow(flow_to_rgb(flow_np))
    axes[0, 3].set_title('Deformation Field', color='#9C27B0')

    # Row 2
    axes[1, 0].imshow(make_checkerboard(f_np, m_np), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Checker: MR / US (Before)')

    axes[1, 1].imshow(make_checkerboard(f_np, w_np), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Checker: MR / Warped (After)')

    axes[1, 2].imshow(make_color_overlay(f_np, w_np))
    axes[1, 2].set_title('Overlay (G=MR, R=Warped)')

    axes[1, 3].imshow(draw_deformation_grid(flow_np))
    axes[1, 3].set_title('Deformation Grid')

    for ax in axes.flat:
        ax.axis('off')

    # 数据来源标签
    src_color = '#1565C0' if source == 'RESECT' else '#E65100'
    ep_str = f'  |  Epoch {epoch}' if epoch is not None else ''

    nmi_delta = nmi_after - nmi_before
    ssim_delta = ssim_after - ssim_before
    fig.suptitle(
        f'[{source}]  Patient {pid}  Slice {sid}{ep_str}\n'
        f'NMI: {nmi_before:.4f} → {nmi_after:.4f} '
        f'({"+" if nmi_delta>=0 else ""}{nmi_delta:.4f})   |   '
        f'SSIM: {ssim_before:.4f} → {ssim_after:.4f} '
        f'({"+" if ssim_delta>=0 else ""}{ssim_delta:.4f})',
        fontsize=13, fontweight='bold', y=0.99, color=src_color
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


# ==================== 验证 ====================

def validate(model, val_loader, criterion, device, epoch, save_dir=None, max_vis=12):
    """验证 + 美化可视化"""
    model.eval()
    losses = []
    nmi_b, nmi_a, ssim_b, ssim_a = [], [], [], []
    resect_nmi, bite_nmi = [], []
    vis_count = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"验证 Ep{epoch+1}", leave=False):
            try:
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

                    source = batch['source'][b]
                    if source == 'RESECT':
                        resect_nmi.append(_nmi_a)
                    else:
                        bite_nmi.append(_nmi_a)

                    if save_dir and vis_count < max_vis:
                        pid = batch['patient_id'][b]
                        sid = batch['slice_idx'][b]
                        src = batch['source'][b]
                        save_beautiful_result(
                            fixed[b], moving[b], warped[b], flow[b:b+1],
                            pid, sid, src, _nmi_b, _nmi_a, _ssim_b, _ssim_a,
                            save_path=os.path.join(save_dir, f"val_{src}_{pid}_S{sid}.png"),
                            epoch=epoch + 1
                        )
                        vis_count += 1

                del fixed, moving, outputs, warped, flow, loss_dict

            except RuntimeError as e:
                print(f"\n  ⚠️ 验证批次出错: {e}, 跳过")
                _empty_cache()
                continue

    # 验证结束后清理内存
    _empty_cache()

    results = {
        'loss': np.mean(losses),
        'nmi_before': np.mean(nmi_b), 'nmi_after': np.mean(nmi_a),
        'ssim_before': np.mean(ssim_b), 'ssim_after': np.mean(ssim_a),
        'nmi_improve': np.mean(nmi_a) - np.mean(nmi_b),
        'ssim_improve': np.mean(ssim_a) - np.mean(ssim_b),
        'resect_nmi': np.mean(resect_nmi) if resect_nmi else 0,
        'bite_nmi': np.mean(bite_nmi) if bite_nmi else 0,
        'num_samples': len(nmi_b)
    }

    print(f"  验证: loss={results['loss']:.4f}, "
          f"NMI {results['nmi_before']:.4f}→{results['nmi_after']:.4f} "
          f"({'↑' if results['nmi_improve']>0 else '↓'}{abs(results['nmi_improve']):.4f}), "
          f"SSIM {results['ssim_before']:.4f}→{results['ssim_after']:.4f} "
          f"({'↑' if results['ssim_improve']>0 else '↓'}{abs(results['ssim_improve']):.4f})")
    if resect_nmi:
        print(f"        RESECT NMI={np.mean(resect_nmi):.4f} ({len(resect_nmi)} samples), "
              f"BITE NMI={np.mean(bite_nmi):.4f} ({len(bite_nmi)} samples)")

    return results


# ==================== 训练 ====================

def _empty_cache():
    """跨平台清理 GPU 缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def _run_one_batch_cpu(model, criterion, fixed_cpu, moving_cpu):
    """在 CPU 上跑一个 batch（OOM 降级用）"""
    model_was_on = next(model.parameters()).device
    model_cpu = model.to('cpu')
    outputs = model_cpu(fixed_cpu, moving_cpu)
    loss_dict = criterion.to('cpu')(fixed_cpu, outputs['warped_lvl0'], outputs['flow_lvl0'])
    loss = loss_dict['total']
    loss.backward()
    # 把模型搬回原设备
    model.to(model_was_on)
    criterion.to(model_was_on)
    return loss.item(), loss_dict


GRAD_ACCUM_STEPS = 4  # 梯度累积步数，等效 batch_size = 1 * 4 = 4


def train(model, train_loader, val_loader, criterion, optimizer, device, config,
          start_epoch=0, best_loss=float('inf'), history=None):
    """训练主循环 (支持断点续训 + 梯度累积 + OOM 降级 CPU)"""
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    for _ in range(start_epoch):
        scheduler.step()

    if history is None:
        history = []

    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        oom_count = 0
        epoch_start = time.time()

        optimizer.zero_grad(set_to_none=True)  # 梯度累积：开头清零

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for step_i, batch in enumerate(progress):
            try:
                fixed = batch['fixed'].to(device)
                moving = batch['moving'].to(device)

                outputs = model(fixed, moving)
                loss_dict = criterion(fixed, outputs['warped_lvl0'], outputs['flow_lvl0'])
                loss = loss_dict['total'] / GRAD_ACCUM_STEPS  # 除以累积步数
                loss.backward()

                epoch_loss += loss.item() * GRAD_ACCUM_STEPS
                batch_count += 1
                oom_count = 0  # 成功则重置 OOM 计数

                _loss_val = loss.item() * GRAD_ACCUM_STEPS
                progress.set_postfix({'loss': f"{_loss_val:.4f}", 'oom': oom_count})

                del fixed, moving, outputs, loss_dict, loss

                # 梯度累积：每 GRAD_ACCUM_STEPS 步做一次 optimizer.step
                if (step_i + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # 定期清理
                if batch_count % 50 == 0:
                    _empty_cache()

            except RuntimeError as e:
                err_msg = str(e).lower()
                if 'out of memory' in err_msg or 'mps' in err_msg:
                    oom_count += 1
                    _empty_cache()

                    # 连续 OOM 5 次，在 CPU 上跑这个 batch
                    if oom_count >= 5 and device.type != 'cpu':
                        try:
                            fixed_cpu = batch['fixed']   # 已在 CPU 上
                            moving_cpu = batch['moving']
                            loss_val, _ = _run_one_batch_cpu(model, criterion,
                                                              fixed_cpu, moving_cpu)
                            epoch_loss += loss_val
                            batch_count += 1
                            oom_count = 0
                            progress.set_postfix({'loss': f"{loss_val:.4f}", 'cpu': True})
                            if (step_i + 1) % GRAD_ACCUM_STEPS == 0:
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                        except Exception as cpu_e:
                            print(f"\n  ⚠️ CPU 降级也失败: {cpu_e}")
                    else:
                        if oom_count <= 3:
                            print(f"\n  ⚠️ OOM (连续{oom_count}次), 跳过...")
                    time.sleep(0.5)
                    continue
                raise

        # epoch 结尾：确保最后的梯度也被 step
        if batch_count % GRAD_ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if batch_count == 0:
            print(f"  Epoch {epoch+1}: 所有批次均失败, 跳过")
            _empty_cache()
            time.sleep(3)
            continue

        avg_train_loss = epoch_loss / batch_count
        scheduler.step()
        epoch_time = time.time() - epoch_start

        _empty_cache()
        time.sleep(1)

        # 每个 epoch 都保存最新检查点 (用于断点续训)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'history': history,
        }, is_best=False, save_path=config['checkpoint_dir'],
           filename='latest_checkpoint.pth')

        # 验证
        val_freq = config.get('val_frequency', 5)
        if (epoch + 1) % val_freq == 0 or epoch == config['num_epochs'] - 1:
            val_results = validate(
                model, val_loader, criterion, device, epoch,
                save_dir=os.path.join(config['result_dir'], f"val_ep{epoch+1}"),
                max_vis=config.get('max_vis', 12)
            )

            history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_results['loss'],
                'nmi_before': val_results['nmi_before'],
                'nmi_after': val_results['nmi_after'],
                'ssim_before': val_results['ssim_before'],
                'ssim_after': val_results['ssim_after'],
                'nmi_improve': val_results['nmi_improve'],
                'ssim_improve': val_results['ssim_improve'],
                'resect_nmi': val_results['resect_nmi'],
                'bite_nmi': val_results['bite_nmi'],
            })

            if val_results['loss'] < best_loss:
                best_loss = val_results['loss']
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'val_results': val_results,
                    'history': history,
                }, is_best=True, save_path=config['checkpoint_dir'])
                print(f"  ✓ 保存最佳模型 (loss={best_loss:.4f})")

            # 保存训练历史 (每次验证后都保存, 防丢失)
            with open(os.path.join(config['result_dir'], 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)

            print(f"  Epoch {epoch+1}: train={avg_train_loss:.4f}, "
                  f"val={val_results['loss']:.4f}, "
                  f"time={epoch_time:.1f}s, lr={scheduler.get_last_lr()[0]:.6f}")
        else:
            print(f"  Epoch {epoch+1}: train={avg_train_loss:.4f}, time={epoch_time:.1f}s")

    # 最终绘图
    if history:
        with open(os.path.join(config['result_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        plot_training_history(history, config['result_dir'])

    return best_loss, history


# ==================== 训练曲线 ====================

def plot_training_history(history, save_dir):
    """美化训练曲线 (含 RESECT/BITE 分项)"""
    matplotlib.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'axes.labelsize': 11, 'legend.fontsize': 9, 'figure.facecolor': 'white'
    })

    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    c_train, c_val = '#1976D2', '#D32F2F'
    c_nmi, c_ssim = '#388E3C', '#7B1FA2'
    c_resect, c_bite = '#0D47A1', '#E65100'

    # ---- Loss ----
    ax = axes[0, 0]
    ax.plot(epochs, [h['train_loss'] for h in history],
            '-o', color=c_train, label='Train Loss', markersize=5, linewidth=2)
    ax.plot(epochs, [h['val_loss'] for h in history],
            '-s', color=c_val, label='Val Loss', markersize=5, linewidth=2)
    ax.fill_between(epochs,
                    [h['train_loss'] for h in history],
                    [h['val_loss'] for h in history], alpha=0.08, color='gray')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(framealpha=0.9); ax.grid(alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ---- NMI Before / After ----
    ax = axes[0, 1]
    ax.plot(epochs, [h['nmi_before'] for h in history],
            '--^', color='gray', label='NMI Before', markersize=4, linewidth=1.5, alpha=0.7)
    ax.plot(epochs, [h['nmi_after'] for h in history],
            '-o', color=c_nmi, label='NMI After', markersize=5, linewidth=2)
    ax.fill_between(epochs, [h['nmi_before'] for h in history],
                    [h['nmi_after'] for h in history], alpha=0.12, color=c_nmi)
    ax.set_xlabel('Epoch'); ax.set_ylabel('NMI')
    ax.set_title('Normalized Mutual Information')
    ax.legend(framealpha=0.9); ax.grid(alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ---- SSIM ----
    ax = axes[1, 0]
    ax.plot(epochs, [h['ssim_before'] for h in history],
            '--^', color='gray', label='SSIM Before', markersize=4, linewidth=1.5, alpha=0.7)
    ax.plot(epochs, [h['ssim_after'] for h in history],
            '-o', color=c_ssim, label='SSIM After', markersize=5, linewidth=2)
    ax.fill_between(epochs, [h['ssim_before'] for h in history],
                    [h['ssim_after'] for h in history], alpha=0.12, color=c_ssim)
    ax.set_xlabel('Epoch'); ax.set_ylabel('SSIM')
    ax.set_title('Structural Similarity Index')
    ax.legend(framealpha=0.9); ax.grid(alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ---- RESECT vs BITE NMI ----
    ax = axes[1, 1]
    resect_nmi = [h['resect_nmi'] for h in history]
    bite_nmi = [h['bite_nmi'] for h in history]
    ax.plot(epochs, resect_nmi, '-o', color=c_resect, label='RESECT NMI', markersize=5, linewidth=2)
    ax.plot(epochs, bite_nmi, '-s', color=c_bite, label='BITE NMI', markersize=5, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('NMI')
    ax.set_title('Per-Dataset NMI Comparison')
    ax.legend(framealpha=0.9); ax.grid(alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('RESECT + BITE Combined Training  |  MR=Fixed, US=Moving',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"训练曲线已保存: {os.path.join(save_dir, 'training_curves.png')}")


def visualize_pairs_overview(pairs, save_path, num_per_source=8):
    """展示 RESECT 和 BITE 配对样本的总览图"""
    resect_pairs = [p for p in pairs if p['source'] == 'RESECT']
    bite_pairs = [p for p in pairs if p['source'] == 'BITE']

    # 从各数据源随机选取样本
    rng = random.Random(42)
    selected = []
    if resect_pairs:
        patients = sorted(set(p['patient_id'] for p in resect_pairs))
        for pid in patients[:num_per_source]:
            pp = [p for p in resect_pairs if p['patient_id'] == pid]
            selected.append(pp[len(pp)//2])
    if bite_pairs:
        patients = sorted(set(p['patient_id'] for p in bite_pairs))
        for pid in patients[:num_per_source]:
            pp = [p for p in bite_pairs if p['patient_id'] == pid]
            selected.append(pp[len(pp)//2])

    n = len(selected)
    cols = min(8, n)
    row_groups = (n + cols - 1) // cols

    fig, axes = plt.subplots(row_groups * 2, cols,
                              figsize=(3 * cols, 3.5 * row_groups * 2))
    if row_groups * 2 == 2:
        axes = axes.reshape(2, cols)

    for i, pair in enumerate(selected):
        rg = (i // cols) * 2
        c = i % cols
        mr_img = np.array(Image.open(pair['mr_path']).convert('L')).astype(np.float32) / 255
        us_img = np.array(Image.open(pair['us_path']).convert('L')).astype(np.float32) / 255

        src_tag = "[R]" if pair['source'] == 'RESECT' else "[B]"
        src_color = '#1565C0' if pair['source'] == 'RESECT' else '#E65100'

        axes[rg, c].imshow(mr_img, cmap='gray', vmin=0, vmax=1)
        axes[rg, c].set_title(f"{src_tag} MR {pair['patient_id']}", fontsize=7, color=src_color, fontweight='bold')
        axes[rg, c].axis('off')

        axes[rg+1, c].imshow(us_img, cmap='gray', vmin=0, vmax=1)
        axes[rg+1, c].set_title(f"US  S{pair['slice_idx']}", fontsize=7, color=src_color)
        axes[rg+1, c].axis('off')

    # 隐藏空白
    total = row_groups * cols
    for i in range(n, total):
        rg = (i // cols) * 2
        c = i % cols
        if rg < row_groups * 2:
            axes[rg, c].axis('off')
            axes[rg+1, c].axis('off')

    fig.suptitle(f'Combined Dataset Overview  |  RESECT ({len(resect_pairs)}) + BITE ({len(bite_pairs)}) pairs',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"配对总览已保存: {save_path}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    RESECT_ROOT = "./resect_png"
    BITE_ROOT   = "./group2 - png"
    OUTPUT_DIR  = "./combined_train_output"
    CHECKPOINT_LOAD = "./bite_train_output/checkpoints/best_model.pth"  # 从 BITE 预训练出发

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"设备: {device}\n")

    # ============ 第一步: 配对两个数据集 ============
    print("=" * 60)
    print("【第一步】配对 RESECT + BITE 数据")
    print("=" * 60 + "\n")

    resect_pairs = pair_resect_dataset(
        RESECT_ROOT,
        content_threshold=0.10,
        max_pairs_per_case=50,
        local_search_window=6,
        mi_threshold=0.02
    )
    bite_pairs = pair_bite_dataset(
        BITE_ROOT,
        content_threshold=0.10,
        mi_threshold=0.03
    )

    all_pairs = resect_pairs + bite_pairs
    random.Random(42).shuffle(all_pairs)

    # 按 source + patient 划分 train/val
    # RESECT: Case1-Case21 训练, Case23-Case27 验证
    # BITE:   01-11 训练, 12-14 验证
    resect_val_cases = {'Case23', 'Case24', 'Case25', 'Case26', 'Case27'}
    bite_val_patients = {'BITE_12', 'BITE_13', 'BITE_14'}

    train_pairs = [p for p in all_pairs
                   if p['patient_id'] not in resect_val_cases
                   and p['patient_id'] not in bite_val_patients]
    val_pairs = [p for p in all_pairs
                 if p['patient_id'] in resect_val_cases
                 or p['patient_id'] in bite_val_patients]

    resect_train = sum(1 for p in train_pairs if p['source'] == 'RESECT')
    bite_train = sum(1 for p in train_pairs if p['source'] == 'BITE')
    resect_val = sum(1 for p in val_pairs if p['source'] == 'RESECT')
    bite_val = sum(1 for p in val_pairs if p['source'] == 'BITE')

    print(f"\n数据划分:")
    print(f"  训练集: {len(train_pairs)} 对  (RESECT={resect_train}, BITE={bite_train})")
    print(f"  验证集: {len(val_pairs)} 对  (RESECT={resect_val}, BITE={bite_val})")

    # 保存配对信息
    pair_info = {
        'total': len(all_pairs),
        'train_count': len(train_pairs),
        'val_count': len(val_pairs),
        'resect_train': resect_train,
        'bite_train': bite_train,
        'resect_val': resect_val,
        'bite_val': bite_val,
    }
    with open(os.path.join(OUTPUT_DIR, 'pair_info.json'), 'w') as f:
        json.dump(pair_info, f, indent=2, ensure_ascii=False)

    # ============ 第二步: 可视化配对 ============
    print("\n【第二步】配对样本可视化\n")
    visualize_pairs_overview(all_pairs,
                              os.path.join(OUTPUT_DIR, 'pair_overview.png'))

    # ============ 第三步: 训练 ============
    print("\n【第三步】联合训练 SelectiveSSM-LTMNet\n")

    # 64x64 大幅减少 GPU 内存占用 (相比 128x128 约省 4 倍)
    IMG_SIZE = 64
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = CombinedPairedDataset(train_pairs, transform=transform)
    val_dataset = CombinedPairedDataset(val_pairs, transform=transform)

    # MPS 不支持 pin_memory
    use_pin = device.type == 'cuda'
    TRAIN_BS = 1   # batch=1，通过梯度累积模拟更大 batch
    VAL_BS = 1
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=True,
                              num_workers=0, pin_memory=use_pin)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BS, shuffle=False,
                            num_workers=0, pin_memory=use_pin)

    print(f"训练: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证: {len(val_dataset)} 样本, {len(val_loader)} 批次")

    # 创建模型
    model = MultiResolutionRegNet(in_channels=2).to(device)
    criterion = DualSimilarityLoss(alpha=10.0, beta=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)  # 微调用较小学习率

    config = {
        'num_epochs': 30,
        'val_frequency': 3,
        'max_vis': 16,             # 每次验证保存的可视化图像数
        'checkpoint_dir': os.path.join(OUTPUT_DIR, 'checkpoints'),
        'result_dir': os.path.join(OUTPUT_DIR, 'results'),
    }
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)

    # 尝试从断点续训
    start_epoch = 0
    best_loss = float('inf')
    saved_history = None
    LATEST_CKPT = os.path.join(config['checkpoint_dir'], 'latest_checkpoint.pth')

    if os.path.isfile(LATEST_CKPT):
        print(f"\n发现断点续训检查点: {LATEST_CKPT}")
        ckpt = torch.load(LATEST_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        saved_history = ckpt.get('history', None)
        print(f"  从 Epoch {start_epoch} 恢复训练 (best_loss={best_loss:.4f})")
    elif os.path.isfile(CHECKPOINT_LOAD):
        print(f"\n加载预训练权重: {CHECKPOINT_LOAD}")
        ckpt = torch.load(CHECKPOINT_LOAD, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        print(f"  已加载 (来自 Epoch {ckpt.get('epoch', '?')}, loss={ckpt.get('best_loss', '?')})")
    else:
        print(f"\n未找到任何检查点，从头训练")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {total_params:,}")

    print(f"\n训练配置:")
    print(f"  Epochs: {config['num_epochs']} (从 Epoch {start_epoch+1} 开始)")
    print(f"  图像尺寸: {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Batch size: {TRAIN_BS} (梯度累积×{GRAD_ACCUM_STEPS}, 等效 batch={TRAIN_BS*GRAD_ACCUM_STEPS})")
    print(f"  学习率: 5e-5 (Cosine 退火, 微调模式)")
    print(f"  损失: NMI + MIND + Smooth (α=10.0, β=0.5)")
    print(f"  验证频率: 每 {config['val_frequency']} epoch")
    print(f"  OOM保护: 连续5次OOM自动降级CPU")
    print(f"  输出: {OUTPUT_DIR}\n")

    start_time = time.time()
    best_loss, history = train(model, train_loader, val_loader, criterion,
                                optimizer, device, config,
                                start_epoch=start_epoch,
                                best_loss=best_loss,
                                history=saved_history)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"联合训练完成!")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  最佳验证损失: {best_loss:.4f}")
    if history:
        best_h = min(history, key=lambda h: h['val_loss'])
        print(f"  最佳 Epoch: {best_h['epoch']}")
        print(f"  整体 NMI: {best_h['nmi_after']:.4f} (Δ={best_h['nmi_improve']:+.4f})")
        print(f"  整体 SSIM: {best_h['ssim_after']:.4f} (Δ={best_h['ssim_improve']:+.4f})")
        print(f"  RESECT NMI: {best_h['resect_nmi']:.4f}")
        print(f"  BITE NMI: {best_h['bite_nmi']:.4f}")
    print(f"{'='*60}")
