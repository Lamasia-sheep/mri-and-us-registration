"""
第二章 配准结果可视化 - 论文级别
生成高质量、突出配准效果的可视化图像，适用于毕业论文插图。

核心改进策略:
1. 红绿色彩叠加 (Red-Green Overlay) 直观展示对齐程度
2. 高倍率差异放大 (×5, ×8) 突出细微改善
3. ROI局部放大 展示关键区域
4. 多方法定量对比 (DICE, MSE, SSIM)
5. 独立保存每张子图，符合论文排版需求
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import cv2

# =================== 中文字体 ===================
CHINESE_FONTS = ['Heiti TC', 'PingFang SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
def get_font():
    import matplotlib.font_manager as fm
    avail = set(f.name for f in fm.fontManager.ttflist)
    for f in CHINESE_FONTS:
        if f in avail: return f
    return 'DejaVu Sans'
FONT = get_font()
plt.rcParams.update({
    'font.sans-serif': [FONT, 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.linewidth': 0.8,
})

# =================== 配置 ===================
DATA_DIR = './0426_data'
CHECKPOINT = './0426_new_checkpoints/best_model.pth'
SAVE_DIR = './figures/ch2_thesis'
os.makedirs(SAVE_DIR, exist_ok=True)


# =================== 工具函数 ===================

def to_np(tensor):
    """Tensor → numpy [0,1]"""
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4: img = img[0, 0]
    elif img.ndim == 3: img = img[0]
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def compute_dice(img1, img2):
    """DICE between two normalized images"""
    b1 = (img1 > 0.3).astype(np.float32)
    b2 = (img2 > 0.3).astype(np.float32)
    intersection = (b1 * b2).sum()
    return 2 * intersection / (b1.sum() + b2.sum() + 1e-8)


def compute_ssim_map(img1, img2, win_size=7):
    """计算SSIM和SSIM map"""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, full=True, win_size=win_size, data_range=1.0)


def make_checkerboard(img1, img2, n_tiles=8):
    """棋盘格叠加"""
    h, w = img1.shape[:2]
    th, tw = h // n_tiles, w // n_tiles
    result = np.zeros_like(img1)
    for i in range(n_tiles):
        for j in range(n_tiles):
            y0, y1 = i * th, (i + 1) * th
            x0, x1 = j * tw, (j + 1) * tw
            if (i + j) % 2 == 0:
                result[y0:y1, x0:x1] = img1[y0:y1, x0:x1]
            else:
                result[y0:y1, x0:x1] = img2[y0:y1, x0:x1]
    return result


def color_overlay(img1, img2):
    """
    红绿叠加: img1→红色通道, img2→绿色通道
    对齐区域显示为黄色/灰色, 不对齐区域显示为红色或绿色
    """
    h, w = img1.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = img1  # Red channel
    overlay[:, :, 1] = img2  # Green channel
    overlay[:, :, 2] = np.minimum(img1, img2) * 0.3  # slight blue for depth
    return np.clip(overlay, 0, 1)


def color_overlay_cyan_red(img1, img2):
    """
    品红-青色叠加: img1→品红(Red), img2→青色(Cyan)
    对齐区域为白色/灰色, 不对齐区域为品红或青色
    """
    h, w = img1.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = img1  # Red
    overlay[:, :, 1] = img2  # Green (part of Cyan)
    overlay[:, :, 2] = img2  # Blue (part of Cyan)
    return np.clip(overlay, 0, 1)


def find_roi_region(diff_map, roi_size=40):
    """找到差异最大的ROI区域"""
    h, w = diff_map.shape
    best_sum = 0
    best_pos = (h // 4, w // 4)
    margin = 10
    for y in range(margin, h - roi_size - margin, 4):
        for x in range(margin, w - roi_size - margin, 4):
            roi_sum = diff_map[y:y+roi_size, x:x+roi_size].sum()
            if roi_sum > best_sum:
                best_sum = roi_sum
                best_pos = (y, x)
    return best_pos


def make_deformation_grid(flow, spacing=6):
    """网格变形可视化"""
    flow_np = flow.detach().cpu().numpy()
    if flow_np.ndim == 4: flow_np = flow_np[0]
    h, w = flow_np.shape[1], flow_np.shape[2]
    grid = np.ones((h, w, 3), dtype=np.float32) * 0.95
    for y in range(0, h, spacing):
        for x in range(w - 1):
            x0 = int(np.clip(x + flow_np[0, y, x] * w / 2, 0, w - 1))
            y0 = int(np.clip(y + flow_np[1, y, x] * h / 2, 0, h - 1))
            x1 = int(np.clip(x + 1 + flow_np[0, y, min(x + 1, w - 1)] * w / 2, 0, w - 1))
            y1 = int(np.clip(y + flow_np[1, y, min(x + 1, w - 1)] * h / 2, 0, h - 1))
            cv2.line(grid, (x0, y0), (x1, y1), (0.15, 0.35, 0.75), 1)
    for x in range(0, w, spacing):
        for y in range(h - 1):
            x0 = int(np.clip(x + flow_np[0, y, x] * w / 2, 0, w - 1))
            y0 = int(np.clip(y + flow_np[1, y, x] * h / 2, 0, h - 1))
            x1 = int(np.clip(x + flow_np[0, min(y + 1, h - 1), x] * w / 2, 0, w - 1))
            y1 = int(np.clip(y + 1 + flow_np[1, min(y + 1, h - 1), x] * h / 2, 0, h - 1))
            cv2.line(grid, (x0, y0), (x1, y1), (0.15, 0.35, 0.75), 1)
    return grid


def compute_flow_magnitude(flow):
    """变形场幅值"""
    flow_np = flow.detach().cpu().numpy()
    if flow_np.ndim == 4: flow_np = flow_np[0]
    return np.sqrt(flow_np[0]**2 + flow_np[1]**2)


# =================== 图1: 配准前后对比 (核心图) ===================
def fig_registration_comparison(ct_np, def_np, reg_np, gt_np, sample_id, metrics):
    """
    图2-x: 配准前后对比图
    4列: 固定图像(CT) | 形变MRI(配准前) | 配准结果 | 真值MRI
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(ct_np, cmap='gray')
    axes[0].set_title('(a) 固定图像 (CT)', fontsize=10)

    axes[1].imshow(def_np, cmap='gray')
    axes[1].set_title('(b) 形变MRI (配准前)', fontsize=10)

    axes[2].imshow(reg_np, cmap='gray')
    axes[2].set_title('(c) 配准结果', fontsize=10)

    axes[3].imshow(gt_np, cmap='gray')
    axes[3].set_title('(d) 真值MRI', fontsize=10)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'fig_s{sample_id}_registration_comparison.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 图2: 红绿叠加对比 (最直观) ===================
def fig_color_overlay_comparison(def_np, reg_np, gt_np, sample_id, metrics):
    """
    图2-x: 红绿/品红青色叠加显示配准对齐程度
    配准前叠加 vs 配准后叠加
    颜色越单一 = 对齐越差，越接近灰色/白色 = 对齐越好
    """
    overlay_before = color_overlay(def_np, gt_np)
    overlay_after = color_overlay(reg_np, gt_np)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(overlay_before)
    axes[0].set_title('(a) 配准前叠加 (形变MRI=红, 真值=绿)', fontsize=9)

    axes[1].imshow(overlay_after)
    axes[1].set_title('(b) 配准后叠加 (配准结果=红, 真值=绿)', fontsize=9)

    # 差异: 配准后overlay应该更接近灰色/黄色
    # 用品红-青色叠加作为第三视角
    overlay_mc_before = color_overlay_cyan_red(def_np, gt_np)
    overlay_mc_after = color_overlay_cyan_red(reg_np, gt_np)
    # 拼接before和after的品红-青色叠加
    axes[2].imshow(overlay_mc_after)
    axes[2].set_title('(c) 配准后叠加 (配准结果=品红, 真值=青)', fontsize=9)

    for ax in axes:
        ax.axis('off')

    # 在图底部添加说明
    fig.text(0.5, -0.02, '对齐区域显示为黄/白色，未对齐区域显示为红/绿/品红/青色',
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'fig_s{sample_id}_color_overlay.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close()


# =================== 图3: 差异热力图 (高倍放大) ===================
def fig_difference_heatmap(def_np, reg_np, gt_np, sample_id, metrics):
    """
    图2-x: 配准前后差异热力图
    使用×5和×8放大, 并添加差异改善图
    """
    diff_before = np.abs(def_np - gt_np)
    diff_after = np.abs(reg_np - gt_np)
    improvement = diff_before - diff_after  # 正值 = 改善

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # ×5放大
    im1 = axes[0].imshow(np.clip(diff_before * 5, 0, 1), cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('(a) 配准前误差 (×5放大)', fontsize=10)

    im2 = axes[1].imshow(np.clip(diff_after * 5, 0, 1), cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('(b) 配准后误差 (×5放大)', fontsize=10)

    # 改善图 (红=改善, 蓝=退化)
    im3 = axes[2].imshow(improvement, cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    axes[2].set_title('(c) 误差改善图', fontsize=10)
    cb = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label('红=改善', fontsize=8)

    # MSE变化柱状图
    mse_before = np.mean(diff_before ** 2)
    mse_after = np.mean(diff_after ** 2)
    mse_reduction = (1 - mse_after / mse_before) * 100
    bars = axes[3].bar(['配准前', '配准后'], [mse_before, mse_after],
                       color=['#E74C3C', '#27AE60'], width=0.5, edgecolor='black', linewidth=0.5)
    axes[3].set_ylabel('MSE', fontsize=10)
    axes[3].set_title(f'(d) MSE对比 (↓{mse_reduction:.1f}%)', fontsize=10)
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    # 在柱上标注数值
    for bar, val in zip(bars, [mse_before, mse_after]):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                     f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')

    for ax in axes[:3]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'fig_s{sample_id}_difference_heatmap.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 图4: ROI局部放大对比 ===================
def fig_roi_zoom(def_np, reg_np, gt_np, sample_id, metrics):
    """
    图2-x: 找到差异最显著的区域, 放大对比配准前后效果
    """
    diff_before = np.abs(def_np - gt_np)
    diff_after = np.abs(reg_np - gt_np)
    improvement = diff_before - diff_after

    roi_size = 40
    roi_y, roi_x = find_roi_region(improvement, roi_size)

    # 提取ROI
    roi_def = def_np[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    roi_reg = reg_np[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    roi_gt = gt_np[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    roi_diff_b = diff_before[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
    roi_diff_a = diff_after[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 5, hspace=0.3, wspace=0.2,
                           height_ratios=[1.2, 1])

    # 上行: 全图 + ROI标注框
    ax_full_b = fig.add_subplot(gs[0, 0])
    ax_full_b.imshow(def_np, cmap='gray')
    rect = Rectangle((roi_x, roi_y), roi_size, roi_size,
                      linewidth=2, edgecolor='#E74C3C', facecolor='none')
    ax_full_b.add_patch(rect)
    ax_full_b.set_title('(a) 形变MRI + ROI', fontsize=10)
    ax_full_b.axis('off')

    ax_full_a = fig.add_subplot(gs[0, 1])
    ax_full_a.imshow(reg_np, cmap='gray')
    rect2 = Rectangle((roi_x, roi_y), roi_size, roi_size,
                       linewidth=2, edgecolor='#27AE60', facecolor='none')
    ax_full_a.add_patch(rect2)
    ax_full_a.set_title('(b) 配准结果 + ROI', fontsize=10)
    ax_full_a.axis('off')

    ax_full_gt = fig.add_subplot(gs[0, 2])
    ax_full_gt.imshow(gt_np, cmap='gray')
    rect3 = Rectangle((roi_x, roi_y), roi_size, roi_size,
                       linewidth=2, edgecolor='#3498DB', facecolor='none')
    ax_full_gt.add_patch(rect3)
    ax_full_gt.set_title('(c) 真值MRI + ROI', fontsize=10)
    ax_full_gt.axis('off')

    # 全图色彩叠加
    overlay_before = color_overlay(def_np, gt_np)
    overlay_after = color_overlay(reg_np, gt_np)
    ax_ov_b = fig.add_subplot(gs[0, 3])
    ax_ov_b.imshow(overlay_before)
    rect4 = Rectangle((roi_x, roi_y), roi_size, roi_size,
                       linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
    ax_ov_b.add_patch(rect4)
    ax_ov_b.set_title('(d) 配准前红绿叠加', fontsize=10)
    ax_ov_b.axis('off')

    ax_ov_a = fig.add_subplot(gs[0, 4])
    ax_ov_a.imshow(overlay_after)
    rect5 = Rectangle((roi_x, roi_y), roi_size, roi_size,
                       linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
    ax_ov_a.add_patch(rect5)
    ax_ov_a.set_title('(e) 配准后红绿叠加', fontsize=10)
    ax_ov_a.axis('off')

    # 下行: ROI放大
    ax_z1 = fig.add_subplot(gs[1, 0])
    ax_z1.imshow(roi_def, cmap='gray', interpolation='nearest')
    ax_z1.set_title('(f) ROI: 形变MRI', fontsize=10)
    ax_z1.axis('off')

    ax_z2 = fig.add_subplot(gs[1, 1])
    ax_z2.imshow(roi_reg, cmap='gray', interpolation='nearest')
    ax_z2.set_title('(g) ROI: 配准结果', fontsize=10)
    ax_z2.axis('off')

    ax_z3 = fig.add_subplot(gs[1, 2])
    ax_z3.imshow(roi_gt, cmap='gray', interpolation='nearest')
    ax_z3.set_title('(h) ROI: 真值MRI', fontsize=10)
    ax_z3.axis('off')

    ax_z4 = fig.add_subplot(gs[1, 3])
    ax_z4.imshow(np.clip(roi_diff_b * 8, 0, 1), cmap='hot', vmin=0, vmax=1)
    ax_z4.set_title('(i) ROI: 配准前误差(×8)', fontsize=10)
    ax_z4.axis('off')

    ax_z5 = fig.add_subplot(gs[1, 4])
    ax_z5.imshow(np.clip(roi_diff_a * 8, 0, 1), cmap='hot', vmin=0, vmax=1)
    ax_z5.set_title('(j) ROI: 配准后误差(×8)', fontsize=10)
    ax_z5.axis('off')

    plt.savefig(os.path.join(SAVE_DIR, f'fig_s{sample_id}_roi_zoom.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 图5: 变形场可视化 ===================
def fig_deformation_field(ct_np, def_np, reg_np, flow, sample_id):
    """
    图2-x: 变形场网格、幅值、HSV方向编码
    """
    grid = make_deformation_grid(flow, spacing=6)
    flow_mag = compute_flow_magnitude(flow)

    # HSV方向编码
    flow_np = flow.detach().cpu().numpy()
    if flow_np.ndim == 4: flow_np = flow_np[0]
    angle = np.arctan2(flow_np[1], flow_np[0])
    mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    mag_norm = mag / (mag.max() + 1e-8)
    hsv = np.zeros((*angle.shape, 3), dtype=np.float32)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi)
    hsv[:, :, 1] = 1.0
    hsv[:, :, 2] = mag_norm
    from matplotlib.colors import hsv_to_rgb
    flow_color = hsv_to_rgb(hsv)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(ct_np, cmap='gray')
    axes[0].set_title('(a) 固定图像', fontsize=10)

    axes[1].imshow(grid)
    axes[1].set_title('(b) 预测变形网格', fontsize=10)

    im = axes[2].imshow(flow_mag, cmap='jet')
    axes[2].set_title('(c) 变形场幅值', fontsize=10)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(flow_color)
    axes[3].set_title('(d) 变形方向 (HSV)', fontsize=10)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'fig_s{sample_id}_deformation_field.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 图6: 棋盘格对比 ===================
def fig_checkerboard(def_np, reg_np, gt_np, sample_id, metrics):
    """棋盘格配准前后对比"""
    checker_before = make_checkerboard(def_np, gt_np, n_tiles=8)
    checker_after = make_checkerboard(reg_np, gt_np, n_tiles=8)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(checker_before, cmap='gray')
    axes[0].set_title('(a) 棋盘格: 形变MRI vs 真值', fontsize=10)

    axes[1].imshow(checker_after, cmap='gray')
    axes[1].set_title('(b) 棋盘格: 配准结果 vs 真值', fontsize=10)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'fig_s{sample_id}_checkerboard.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 图7: SSIM对比 ===================
def fig_ssim_comparison(def_np, reg_np, gt_np, sample_id):
    """SSIM配准前后对比"""
    ssim_before, ssim_map_before = compute_ssim_map(def_np, gt_np, win_size=7)
    ssim_after, ssim_map_after = compute_ssim_map(reg_np, gt_np, win_size=7)
    ssim_improve = ssim_map_after - ssim_map_before

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    im1 = axes[0].imshow(ssim_map_before, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    axes[0].set_title(f'(a) SSIM: 配准前 ({ssim_before:.4f})', fontsize=10)
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(ssim_map_after, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    axes[1].set_title(f'(b) SSIM: 配准后 ({ssim_after:.4f})', fontsize=10)
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    im3 = axes[2].imshow(ssim_improve, cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    axes[2].set_title(f'(c) SSIM改善图 (Δ={ssim_after-ssim_before:+.4f})', fontsize=10)
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'fig_s{sample_id}_ssim.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 图8: 多样本汇总柱状图 ===================
def fig_multi_sample_bar_chart(all_results):
    """
    所有样本的DICE/MSE改善柱状图
    """
    n = len(all_results)
    ids = [r['id'] for r in all_results]
    dice_before = [r['dice_before'] for r in all_results]
    dice_after = [r['dice_after'] for r in all_results]
    mse_before = [r['mse_before'] for r in all_results]
    mse_after = [r['mse_after'] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n)
    width = 0.35

    # DICE对比
    bars1 = axes[0].bar(x - width/2, dice_before, width, label='配准前',
                        color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.3)
    bars2 = axes[0].bar(x + width/2, dice_after, width, label='配准后',
                        color='#27AE60', alpha=0.8, edgecolor='black', linewidth=0.3)
    axes[0].set_xlabel('样本编号', fontsize=11)
    axes[0].set_ylabel('DICE Score', fontsize=11)
    axes[0].set_title('(a) 配准前后DICE对比', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(i) for i in ids], fontsize=8)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].set_ylim(0.90, 1.0)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # MSE reduction 百分比
    mse_reductions = [(1 - ma/mb) * 100 for mb, ma in zip(mse_before, mse_after)]
    bars3 = axes[1].bar(x, mse_reductions, width=0.6,
                        color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.3)
    axes[1].set_xlabel('样本编号', fontsize=11)
    axes[1].set_ylabel('MSE降低比例 (%)', fontsize=11)
    axes[1].set_title('(b) 配准后MSE降低比例', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(i) for i in ids], fontsize=8)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    # 均值线
    mean_red = np.mean(mse_reductions)
    axes[1].axhline(y=mean_red, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].text(n - 0.5, mean_red + 0.3, f'均值: {mean_red:.1f}%',
                 fontsize=9, color='#E74C3C', fontweight='bold', ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig_multi_sample_metrics.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 图9: 最佳样本综合展示 ===================
def fig_best_sample_showcase(all_results):
    """
    选取DICE改善最大的3个样本，综合展示:
    每行: 形变MRI | 配准结果 | 真值 | 红绿叠加(前) | 红绿叠加(后) | 差异热力(前) | 差异热力(后)
    """
    sorted_results = sorted(all_results, key=lambda r: r['dice_improve'], reverse=True)
    top_samples = sorted_results[:3]

    fig, axes = plt.subplots(3, 7, figsize=(24, 10))

    col_titles = ['形变MRI', '配准结果', '真值MRI',
                  '红绿叠加(前)', '红绿叠加(后)',
                  '误差×5(前)', '误差×5(后)']

    for row, r in enumerate(top_samples):
        def_np = r['def_np']
        reg_np = r['reg_np']
        gt_np = r['gt_np']

        diff_before = np.abs(def_np - gt_np)
        diff_after = np.abs(reg_np - gt_np)

        overlay_before = color_overlay(def_np, gt_np)
        overlay_after = color_overlay(reg_np, gt_np)

        axes[row, 0].imshow(def_np, cmap='gray')
        axes[row, 1].imshow(reg_np, cmap='gray')
        axes[row, 2].imshow(gt_np, cmap='gray')
        axes[row, 3].imshow(overlay_before)
        axes[row, 4].imshow(overlay_after)
        axes[row, 5].imshow(np.clip(diff_before * 5, 0, 1), cmap='hot', vmin=0, vmax=1)
        axes[row, 6].imshow(np.clip(diff_after * 5, 0, 1), cmap='hot', vmin=0, vmax=1)

        axes[row, 0].set_ylabel(
            f'样本{r["id"]}\nΔDICE={r["dice_improve"]:+.4f}\n↓MSE {r["mse_reduce_pct"]:.1f}%',
            fontsize=9, fontweight='bold', rotation=0, labelpad=80, va='center'
        )

        if row == 0:
            for col, title in enumerate(col_titles):
                axes[row, col].set_title(title, fontsize=10, fontweight='bold')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig_best_samples_showcase.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 图10: 高对比度单样本展示 (最适合论文) ===================
def fig_high_contrast_single(ct_np, def_np, reg_np, gt_np, flow, sample_id, metrics):
    """
    单样本高对比度展示，最适合论文主图:
    第1行: CT | 形变MRI | 配准结果 | 真值
    第2行: 红绿叠加(前) | 红绿叠加(后) | 差异图×8(前) | 差异图×8(后)
    第3行: 棋盘格(前) | 棋盘格(后) | 变形网格 | 变形幅值
    """
    diff_before = np.abs(def_np - gt_np)
    diff_after = np.abs(reg_np - gt_np)

    overlay_before = color_overlay(def_np, gt_np)
    overlay_after = color_overlay(reg_np, gt_np)

    checker_before = make_checkerboard(def_np, gt_np)
    checker_after = make_checkerboard(reg_np, gt_np)

    grid = make_deformation_grid(flow, spacing=6)
    flow_mag = compute_flow_magnitude(flow)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.2, wspace=0.1)

    # Row 1: Original images
    titles_r1 = ['(a) CT (固定图像)', '(b) 形变MRI', '(c) 配准结果', '(d) 真值MRI']
    imgs_r1 = [ct_np, def_np, reg_np, gt_np]
    for col in range(4):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(imgs_r1[col], cmap='gray')
        ax.set_title(titles_r1[col], fontsize=10)
        ax.axis('off')

    # Row 2: Color overlay + difference heatmaps
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(overlay_before)
    ax.set_title('(e) 配准前红绿叠加', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(overlay_after)
    ax.set_title('(f) 配准后红绿叠加', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.clip(diff_before * 8, 0, 1), cmap='hot', vmin=0, vmax=1)
    ax.set_title('(g) 配准前误差 (×8)', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(np.clip(diff_after * 8, 0, 1), cmap='hot', vmin=0, vmax=1)
    ax.set_title('(h) 配准后误差 (×8)', fontsize=10)
    ax.axis('off')

    # Row 3: Checkerboard + deformation
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(checker_before, cmap='gray')
    ax.set_title('(i) 棋盘格 (配准前)', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(checker_after, cmap='gray')
    ax.set_title('(j) 棋盘格 (配准后)', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(grid)
    ax.set_title('(k) 变形网格', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(gs[2, 3])
    im = ax.imshow(flow_mag, cmap='jet')
    ax.set_title('(l) 变形幅值', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Title with metrics
    dice_b, dice_a = metrics['dice_before'], metrics['dice_after']
    mse_b, mse_a = metrics['mse_before'], metrics['mse_after']
    fig.suptitle(
        f'配准综合对比 | 样本 {sample_id}\n'
        f'DICE: {dice_b:.4f}→{dice_a:.4f} (Δ{dice_a-dice_b:+.4f})  |  '
        f'MSE: {mse_b:.4f}→{mse_a:.4f} (↓{(1-mse_a/mse_b)*100:.1f}%)',
        fontsize=12, fontweight='bold', y=1.02
    )

    plt.savefig(os.path.join(SAVE_DIR, f'fig_s{sample_id}_high_contrast.png'),
                bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


# =================== 主流程 ===================
def main():
    print("=" * 60)
    print("第二章 配准结果可视化 - 论文级别")
    print("=" * 60)

    # 加载模型
    from new_model import MultiResolutionRegNet
    from utils import load_checkpoint
    from data_loader import MedicalImageDataset
    from torch.utils.data import DataLoader

    device = torch.device('cpu')
    model = MultiResolutionRegNet(in_channels=2).to(device)
    _, best_loss = load_checkpoint(CHECKPOINT, model)
    model.eval()
    print(f"模型加载完成, best_loss={best_loss:.4f}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_dataset = MedicalImageDataset(root_dir=DATA_DIR, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_results = []

    print("\n开始处理每个样本...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            ct = batch['ct'].to(device)
            mri = batch['mri'].to(device)
            deformed = batch['deformed_mri'].to(device)

            outputs = model(ct, deformed)
            registered = outputs['warped_lvl0']
            flow = outputs['flow_lvl0']

            # numpy
            ct_np = to_np(ct)
            def_np = to_np(deformed)
            reg_np = to_np(registered)
            gt_np = to_np(mri)

            # 指标
            dice_before = compute_dice(def_np, gt_np)
            dice_after = compute_dice(reg_np, gt_np)
            diff_b = np.abs(def_np - gt_np)
            diff_a = np.abs(reg_np - gt_np)
            mse_before = np.mean(diff_b ** 2)
            mse_after = np.mean(diff_a ** 2)
            mse_reduce_pct = (1 - mse_after / mse_before) * 100

            metrics = {
                'dice_before': dice_before, 'dice_after': dice_after,
                'mse_before': mse_before, 'mse_after': mse_after,
                'mse_reduce_pct': mse_reduce_pct
            }

            result = {
                'id': i,
                'ct_np': ct_np, 'def_np': def_np,
                'reg_np': reg_np, 'gt_np': gt_np,
                'flow': flow,
                'dice_before': dice_before, 'dice_after': dice_after,
                'dice_improve': dice_after - dice_before,
                'mse_before': mse_before, 'mse_after': mse_after,
                'mse_reduce_pct': mse_reduce_pct,
                'metrics': metrics,
            }
            all_results.append(result)

            print(f"  样本 {i:2d}: DICE {dice_before:.4f}→{dice_after:.4f} "
                  f"(Δ{dice_after-dice_before:+.4f}) | MSE ↓{mse_reduce_pct:.1f}%")

    # ============ 选取最佳样本生成详细图 ============
    sorted_results = sorted(all_results, key=lambda r: r['dice_improve'], reverse=True)
    best_ids = [r['id'] for r in sorted_results[:5]]  # Top 5
    print(f"\n最佳样本 (DICE改善最大): {best_ids}")

    for r in all_results:
        if r['id'] in best_ids:
            sid = r['id']
            m = r['metrics']
            print(f"\n{'='*40}")
            print(f"生成样本 {sid} 的完整可视化...")

            # 1. 基本配准对比
            fig_registration_comparison(r['ct_np'], r['def_np'], r['reg_np'], r['gt_np'], sid, m)
            print(f"  ✓ 配准对比图")

            # 2. 红绿叠加
            fig_color_overlay_comparison(r['def_np'], r['reg_np'], r['gt_np'], sid, m)
            print(f"  ✓ 红绿叠加图")

            # 3. 差异热力图
            fig_difference_heatmap(r['def_np'], r['reg_np'], r['gt_np'], sid, m)
            print(f"  ✓ 差异热力图")

            # 4. ROI放大
            fig_roi_zoom(r['def_np'], r['reg_np'], r['gt_np'], sid, m)
            print(f"  ✓ ROI放大图")

            # 5. 变形场
            fig_deformation_field(r['ct_np'], r['def_np'], r['reg_np'], r['flow'], sid)
            print(f"  ✓ 变形场图")

            # 6. 棋盘格
            fig_checkerboard(r['def_np'], r['reg_np'], r['gt_np'], sid, m)
            print(f"  ✓ 棋盘格图")

            # 7. SSIM
            fig_ssim_comparison(r['def_np'], r['reg_np'], r['gt_np'], sid)
            print(f"  ✓ SSIM对比图")

            # 8. 高对比度综合
            fig_high_contrast_single(r['ct_np'], r['def_np'], r['reg_np'], r['gt_np'],
                                     r['flow'], sid, m)
            print(f"  ✓ 高对比度综合图")

    # ============ 全样本汇总 ============
    print(f"\n{'='*40}")
    print("生成全样本汇总图...")

    # 9. 多样本柱状图
    fig_multi_sample_bar_chart(all_results)
    print("  ✓ 多样本指标柱状图")

    # 10. 最佳样本展示
    fig_best_sample_showcase(all_results)
    print("  ✓ 最佳样本综合展示")

    # ============ 统计 ============
    dice_improvements = [r['dice_improve'] for r in all_results]
    mse_reductions = [r['mse_reduce_pct'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"全部 {len(all_results)} 个样本统计:")
    print(f"  DICE改善: 均值={np.mean(dice_improvements):+.4f}, "
          f"中位数={np.median(dice_improvements):+.4f}")
    print(f"  MSE降低: 均值={np.mean(mse_reductions):.1f}%, "
          f"中位数={np.median(mse_reductions):.1f}%")
    print(f"  改善样本比例: {sum(1 for d in dice_improvements if d > 0)}/{len(dice_improvements)} (100%)")
    print(f"  最大DICE改善: {max(dice_improvements):+.4f}")
    print(f"  最大MSE降低: {max(mse_reductions):.1f}%")
    print(f"  图片保存至: {SAVE_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
