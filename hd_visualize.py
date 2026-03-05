"""
高清推理 + 论文级可视化脚本
================================
用训好的 best_model.pth 在 256×256 分辨率下推理,
生成论文可直接使用的高清配准结果图。

用法:
    python hd_visualize.py
"""

import os
import gc
import random
import numpy as np
from PIL import Image
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from new_model import MultiResolutionRegNet

# ==================== 配置 ====================

CHECKPOINT = "./combined_train_output/checkpoints/best_model.pth"
RESECT_ROOT = "./resect_png"
BITE_ROOT = "./group2 - png"
OUTPUT_DIR = "./hd_results"

IMG_SIZE = 256          # 高清推理分辨率
MAX_SAMPLES = 30        # 最多生成多少张
DPI = 300               # 论文印刷级 DPI

# ==================== 工具函数 ====================

def compute_content_ratio(img_path, threshold=15):
    img = np.array(Image.open(img_path).convert('L'))
    return (img > threshold).sum() / img.size


def compute_nmi(x, y, bins=64):
    """NMI (numpy arrays or tensors)"""
    if hasattr(x, 'cpu'):
        x = x.detach().cpu().numpy().flatten()
    else:
        x = x.flatten()
    if hasattr(y, 'cpu'):
        y = y.detach().cpu().numpy().flatten()
    else:
        y = y.flatten()
    h2d, _, _ = np.histogram2d(x, y, bins=bins)
    h2d = h2d / (h2d.sum() + 1e-10)
    px, py = h2d.sum(1), h2d.sum(0)
    px_py = px[:, None] * py[None, :]
    nz = h2d > 0
    mi = (h2d[nz] * np.log(h2d[nz] / (px_py[nz] + 1e-10))).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return 2 * mi / (hx + hy + 1e-10)


def compute_ssim_np(img1, img2):
    """SSIM (numpy, 0~1 range)"""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, data_range=1.0)


def to_np(t):
    """tensor → 0~1 numpy"""
    x = t.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    return x.astype(np.float32)


# ==================== 可视化子函数 ====================

def flow_to_rgb(flow_np):
    """(2,H,W) → HSV 彩色图"""
    u, v = flow_np[0], flow_np[1]
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    max_mag = mag.max() if mag.max() > 0 else 1.0
    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def flow_magnitude_map(flow_np):
    """(2,H,W) → 形变幅度热力图"""
    mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    return mag


def make_checkerboard(img_a, img_b, block=8):
    H, W = img_a.shape[:2]
    board = np.zeros((H, W), dtype=bool)
    for i in range(0, H, block):
        for j in range(0, W, block):
            if ((i // block) + (j // block)) % 2 == 0:
                board[i:i+block, j:j+block] = True
    return np.where(board, img_a, img_b)


def make_color_overlay(fixed_np, warped_np):
    H, W = fixed_np.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[..., 1] = fixed_np           # 绿色 = MR
    overlay[..., 0] = warped_np          # 红色 = Warped US
    overlay[..., 2] = (fixed_np * 0.3 + warped_np * 0.3)  # 蓝色通道淡化
    return np.clip(overlay, 0, 1)


def make_diff_map(img_a, img_b):
    """差异热力图"""
    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    return diff


def draw_deformation_grid(flow_np, grid_step=8):
    u, v = flow_np[0], flow_np[1]
    H, W = u.shape
    canvas = np.ones((H, W, 3), dtype=np.float32)
    for i in range(0, H, grid_step):
        for j in range(W - 1):
            y1 = int(np.clip(i + v[i, j], 0, H-1))
            x1 = int(np.clip(j + u[i, j], 0, W-1))
            y2 = int(np.clip(i + v[i, j+1], 0, H-1))
            x2 = int(np.clip(j+1 + u[i, j+1], 0, W-1))
            cv2.line(canvas, (x1, y1), (x2, y2), (0.15, 0.35, 0.75), 1, cv2.LINE_AA)
    for j in range(0, W, grid_step):
        for i in range(H - 1):
            y1 = int(np.clip(i + v[i, j], 0, H-1))
            x1 = int(np.clip(j + u[i, j], 0, W-1))
            y2 = int(np.clip(i+1 + v[i+1, j], 0, H-1))
            x2 = int(np.clip(j + u[i+1, j], 0, W-1))
            cv2.line(canvas, (x1, y1), (x2, y2), (0.15, 0.35, 0.75), 1, cv2.LINE_AA)
    return canvas


# ==================== 论文级大图可视化 ====================

def save_hd_result(fixed_t, moving_t, warped_t, flow_t,
                   patient_id, slice_idx, source, save_path):
    """
    3 行 × 4 列, 论文可直接使用的高清配准结果图
    
    Row 1: MR (Fixed)     |  US (Moving)       |  Warped US          |  |Warped - MR| diff
    Row 2: Checker Before |  Checker After      |  Color Overlay      |  Deformation Grid
    Row 3: Flow HSV       |  Flow Magnitude     |  Diff Before→After  |  Metrics Summary
    """
    matplotlib.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'font.family': 'DejaVu Sans'
    })

    f_np = to_np(fixed_t)
    m_np = to_np(moving_t)
    w_np = to_np(warped_t)
    flow_np = flow_t.detach().cpu().numpy()
    if flow_np.ndim == 4:
        flow_np = flow_np[0]

    # 计算指标
    nmi_before = compute_nmi(f_np, m_np)
    nmi_after = compute_nmi(f_np, w_np)
    try:
        ssim_before = compute_ssim_np(f_np, m_np)
        ssim_after = compute_ssim_np(f_np, w_np)
    except ImportError:
        ssim_before = ssim_after = 0.0

    fig = plt.figure(figsize=(24, 18), facecolor='white')
    gs = GridSpec(3, 4, figure=fig, hspace=0.22, wspace=0.08)

    def add_img(r, c, img, title, cmap='gray', vmin=0, vmax=1, colorbar=False):
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax.set_title(title, pad=8)
        ax.axis('off')
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.2f')
        return ax

    # ---- Row 1 ----
    add_img(0, 0, f_np, 'MR (Fixed)', cmap='gray')
    add_img(0, 1, m_np, 'US (Moving)', cmap='gray')
    add_img(0, 2, w_np, 'Warped US (Result)', cmap='gray')

    diff_after = make_diff_map(f_np, w_np)
    add_img(0, 3, diff_after, '|MR − Warped| Difference', cmap='hot', vmin=0, 
            vmax=max(diff_after.max(), 0.3), colorbar=True)

    # ---- Row 2 ----
    add_img(1, 0, make_checkerboard(f_np, m_np, block=12), 'Checkerboard: Before')
    add_img(1, 1, make_checkerboard(f_np, w_np, block=12), 'Checkerboard: After')

    ax_overlay = fig.add_subplot(gs[1, 2])
    ax_overlay.imshow(make_color_overlay(f_np, w_np), interpolation='bilinear')
    ax_overlay.set_title('Color Overlay\n(Green=MR, Red=Warped)', pad=8)
    ax_overlay.axis('off')

    ax_grid = fig.add_subplot(gs[1, 3])
    ax_grid.imshow(draw_deformation_grid(flow_np, grid_step=10), interpolation='bilinear')
    ax_grid.set_title('Deformation Grid', pad=8)
    ax_grid.axis('off')

    # ---- Row 3 ----
    ax_flow = fig.add_subplot(gs[2, 0])
    ax_flow.imshow(flow_to_rgb(flow_np), interpolation='bilinear')
    ax_flow.set_title('Flow Field (HSV)', pad=8)
    ax_flow.axis('off')

    mag = flow_magnitude_map(flow_np)
    add_img(2, 1, mag, 'Flow Magnitude', cmap='magma', vmin=0,
            vmax=max(mag.max(), 1.0), colorbar=True)

    # 配准前后差异对比
    diff_before = make_diff_map(f_np, m_np)
    ax_diff = fig.add_subplot(gs[2, 2])
    comb = np.concatenate([diff_before, np.ones((f_np.shape[0], 4)) * 0.5, diff_after], axis=1)
    ax_diff.imshow(comb, cmap='hot', vmin=0, vmax=max(diff_before.max(), diff_after.max(), 0.3),
                   interpolation='bilinear')
    mid = diff_before.shape[1]
    ax_diff.axvline(x=mid + 2, color='white', linewidth=2, linestyle='--')
    ax_diff.set_title('Diff: Before (L) → After (R)', pad=8)
    ax_diff.axis('off')

    # ---- 指标卡片 ----
    ax_card = fig.add_subplot(gs[2, 3])
    ax_card.axis('off')

    src_color = '#1565C0' if source == 'RESECT' else '#E65100'
    nmi_delta = nmi_after - nmi_before
    ssim_delta = ssim_after - ssim_before

    card_text = (
        f"{'━' * 32}\n"
        f"  📊  Registration Metrics\n"
        f"{'━' * 32}\n\n"
        f"  NMI   Before:  {nmi_before:.4f}\n"
        f"  NMI   After:   {nmi_after:.4f}\n"
        f"  NMI   Δ:       {'↑' if nmi_delta>=0 else '↓'}{abs(nmi_delta):.4f}\n\n"
        f"  SSIM  Before:  {ssim_before:.4f}\n"
        f"  SSIM  After:   {ssim_after:.4f}\n"
        f"  SSIM  Δ:       {'↑' if ssim_delta>=0 else '↓'}{abs(ssim_delta):.4f}\n\n"
        f"{'━' * 32}\n"
        f"  Resolution:  {IMG_SIZE}×{IMG_SIZE}\n"
        f"  Dataset:     {source}\n"
        f"  Patient:     {patient_id}\n"
        f"  Slice:       {slice_idx}\n"
        f"{'━' * 32}"
    )
    ax_card.text(0.05, 0.95, card_text, transform=ax_card.transAxes,
                 fontsize=12, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                           edgecolor='#BDBDBD', alpha=0.95))

    # 总标题
    fig.suptitle(
        f'[{source}]  Patient {patient_id}  —  Slice {slice_idx}   |   '
        f'NMI: {nmi_before:.4f} → {nmi_after:.4f}  ({"+" if nmi_delta>=0 else ""}{nmi_delta:.4f})',
        fontsize=16, fontweight='bold', y=0.98, color=src_color
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


# ==================== 采集样本 ====================

def collect_samples(resect_root, bite_root, max_total=30):
    """从两个数据集各采一些质量好的样本"""
    samples = []

    # RESECT 样本
    if os.path.isdir(resect_root):
        cases = sorted([d for d in os.listdir(resect_root)
                        if d.startswith('Case') and os.path.isdir(os.path.join(resect_root, d))])
        rng = random.Random(42)
        for case in cases[:15]:  # 最多 15 个 case
            t1_dir = os.path.join(resect_root, case, 'MRI_T1')
            us_dir = os.path.join(resect_root, case, 'US_before')
            if not os.path.isdir(t1_dir) or not os.path.isdir(us_dir):
                continue
            t1_files = sorted([f for f in os.listdir(t1_dir) if f.endswith('.png')])
            us_files = sorted([f for f in os.listdir(us_dir) if f.endswith('.png')])
            if not t1_files or not us_files:
                continue
            # 取中间切片 (解剖最丰富)
            t1_mid = t1_files[len(t1_files) // 2]
            us_mid = us_files[len(us_files) // 2]
            samples.append({
                'mr_path': os.path.join(t1_dir, t1_mid),
                'us_path': os.path.join(us_dir, us_mid),
                'patient_id': case,
                'slice_idx': f"T{t1_mid.replace('.png','')}_U{us_mid.replace('.png','')}",
                'source': 'RESECT'
            })

    # BITE 样本
    if os.path.isdir(bite_root):
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
            mr_files = {int(f.replace('.png', '')): f for f in os.listdir(mr_dir) if f.endswith('.png')}
            us_files = {int(f.replace('.png', '')): f for f in os.listdir(us_dir) if f.endswith('.png')}
            common = sorted(set(mr_files) & set(us_files))
            if not common:
                continue
            # 取中间高质量切片
            mid_idx = common[len(common) // 2]
            samples.append({
                'mr_path': os.path.join(mr_dir, mr_files[mid_idx]),
                'us_path': os.path.join(us_dir, us_files[mid_idx]),
                'patient_id': f"BITE_{pid}",
                'slice_idx': mid_idx,
                'source': 'BITE'
            })

    rng = random.Random(42)
    rng.shuffle(samples)
    return samples[:max_total]


# ==================== 主程序 ====================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"推理分辨率: {IMG_SIZE}×{IMG_SIZE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"DPI: {DPI}\n")

    # 加载模型
    model = MultiResolutionRegNet(in_channels=2).to(device)
    if not os.path.isfile(CHECKPOINT):
        print(f"❌ 找不到检查点: {CHECKPOINT}")
        print("   请先完成训练 (python combined_train.py)")
        exit(1)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"✅ 模型加载成功 (Epoch {ckpt.get('epoch', '?')}, loss={ckpt.get('best_loss', '?')})\n")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 采集样本
    samples = collect_samples(RESECT_ROOT, BITE_ROOT, max_total=MAX_SAMPLES)
    print(f"采集到 {len(samples)} 个样本\n")

    # 推理 + 可视化
    resect_nmi_b, resect_nmi_a = [], []
    bite_nmi_b, bite_nmi_a = [], []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc="高清推理")):
            mr_img = Image.open(sample['mr_path']).convert('L')
            us_img = Image.open(sample['us_path']).convert('L')

            mr_t = transform(mr_img).unsqueeze(0).to(device)
            us_t = transform(us_img).unsqueeze(0).to(device)

            outputs = model(mr_t, us_t)
            warped = outputs['warped_lvl0']
            flow = outputs['flow_lvl0']

            pid = sample['patient_id']
            sid = sample['slice_idx']
            src = sample['source']

            save_path = os.path.join(OUTPUT_DIR, f"{i:03d}_{src}_{pid}_S{sid}.png")
            save_hd_result(mr_t[0], us_t[0], warped[0], flow[0:1],
                           pid, sid, src, save_path)

            # 统计
            f_np = to_np(mr_t[0])
            w_np = to_np(warped[0])
            m_np = to_np(us_t[0])
            nb = compute_nmi(f_np, m_np)
            na = compute_nmi(f_np, w_np)
            if src == 'RESECT':
                resect_nmi_b.append(nb)
                resect_nmi_a.append(na)
            else:
                bite_nmi_b.append(nb)
                bite_nmi_a.append(na)

            del mr_t, us_t, outputs, warped, flow
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'='*50}")
    print(f"高清可视化完成! 共 {len(samples)} 张图已保存到 {OUTPUT_DIR}/")
    print(f"{'='*50}")
    if resect_nmi_b:
        print(f"  RESECT ({len(resect_nmi_b)} samples):")
        print(f"    NMI: {np.mean(resect_nmi_b):.4f} → {np.mean(resect_nmi_a):.4f} "
              f"(Δ={np.mean(resect_nmi_a)-np.mean(resect_nmi_b):+.4f})")
    if bite_nmi_b:
        print(f"  BITE ({len(bite_nmi_b)} samples):")
        print(f"    NMI: {np.mean(bite_nmi_b):.4f} → {np.mean(bite_nmi_a):.4f} "
              f"(Δ={np.mean(bite_nmi_a)-np.mean(bite_nmi_b):+.4f})")
    print(f"{'='*50}")
