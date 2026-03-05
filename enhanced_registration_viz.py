"""
增强配准结果可视化
用差异图、棋盘格、网格变形、边缘叠加等方式让配准过程和结果更加直观突出
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import cv2

# =================== 配置 ===================
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
    'font.size': 10,
})

DATA_DIR = './0426_data'
CHECKPOINT = './0426_new_checkpoints/best_model.pth'
SAVE_DIR = './figures/registration_enhanced'
os.makedirs(SAVE_DIR, exist_ok=True)


# =================== 工具函数 ===================

def load_model():
    """加载训练好的模型"""
    from new_model import MultiResolutionRegNet
    from utils import load_checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiResolutionRegNet(in_channels=2).to(device)
    _, best_loss = load_checkpoint(CHECKPOINT, model)
    model.eval()
    print(f"模型加载完成, best_loss={best_loss:.4f}, device={device}")
    return model, device


def load_test_data(device):
    """加载测试数据"""
    from data_loader import MedicalImageDataset
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_dataset = MedicalImageDataset(root_dir=DATA_DIR, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    return test_loader


def to_np(tensor):
    """Tensor → numpy [0,1]"""
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0, 0]
    elif img.ndim == 3:
        img = img[0]
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def compute_local_ssim_map(img1, img2, win_size=7):
    """计算局部SSIM map"""
    from skimage.metrics import structural_similarity
    ssim_val, ssim_map = structural_similarity(img1, img2, full=True, win_size=win_size, data_range=1.0)
    return ssim_val, ssim_map


def make_checkerboard(img1, img2, n_tiles=8):
    """生成棋盘格叠加图"""
    h, w = img1.shape[:2]
    tile_h, tile_w = h // n_tiles, w // n_tiles
    result = np.zeros_like(img1)
    for i in range(n_tiles):
        for j in range(n_tiles):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            if (i + j) % 2 == 0:
                result[y0:y1, x0:x1] = img1[y0:y1, x0:x1]
            else:
                result[y0:y1, x0:x1] = img2[y0:y1, x0:x1]
    return result


def make_edge_overlay(base_img, ref_img, color=(1, 0.2, 0.2)):
    """在base_img上叠加ref_img的边缘"""
    ref_uint8 = (ref_img * 255).astype(np.uint8)
    edges = cv2.Canny(ref_uint8, 30, 100)
    edges = edges.astype(np.float32) / 255.0
    
    # 转为RGB
    if base_img.ndim == 2:
        rgb = np.stack([base_img] * 3, axis=-1)
    else:
        rgb = base_img.copy()
    
    for c in range(3):
        rgb[:, :, c] = rgb[:, :, c] * (1 - edges * 0.7) + color[c] * edges * 0.7
    return np.clip(rgb, 0, 1)


def make_deformation_grid(flow, spacing=8):
    """生成网格变形可视化"""
    flow_np = flow.detach().cpu().numpy()
    if flow_np.ndim == 4:
        flow_np = flow_np[0]
    
    h, w = flow_np.shape[1], flow_np.shape[2]
    grid = np.ones((h, w, 3), dtype=np.float32) * 0.95
    
    # 绘制变形后的网格线
    # 水平线
    for y in range(0, h, spacing):
        for x in range(w - 1):
            x0 = int(np.clip(x + flow_np[0, y, x] * w / 2, 0, w - 1))
            y0 = int(np.clip(y + flow_np[1, y, x] * h / 2, 0, h - 1))
            x1 = int(np.clip(x + 1 + flow_np[0, y, min(x + 1, w - 1)] * w / 2, 0, w - 1))
            y1 = int(np.clip(y + flow_np[1, y, min(x + 1, w - 1)] * h / 2, 0, h - 1))
            cv2.line(grid, (x0, y0), (x1, y1), (0.2, 0.4, 0.8), 1)
    
    # 垂直线
    for x in range(0, w, spacing):
        for y in range(h - 1):
            x0 = int(np.clip(x + flow_np[0, y, x] * w / 2, 0, w - 1))
            y0 = int(np.clip(y + flow_np[1, y, x] * h / 2, 0, h - 1))
            x1 = int(np.clip(x + flow_np[0, min(y + 1, h - 1), x] * w / 2, 0, w - 1))
            y1 = int(np.clip(y + 1 + flow_np[1, min(y + 1, h - 1), x] * h / 2, 0, h - 1))
            cv2.line(grid, (x0, y0), (x1, y1), (0.2, 0.4, 0.8), 1)
    
    return grid


def compute_flow_magnitude(flow):
    """计算变形场幅值"""
    flow_np = flow.detach().cpu().numpy()
    if flow_np.ndim == 4:
        flow_np = flow_np[0]
    mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)
    return mag


def compute_dice(pred, target):
    """计算DICE"""
    pred_np = to_np(pred)
    tgt_np = to_np(target)
    pred_b = (pred_np > 0.3).astype(np.float32)
    tgt_b = (tgt_np > 0.3).astype(np.float32)
    intersection = (pred_b * tgt_b).sum()
    return 2 * intersection / (pred_b.sum() + tgt_b.sum() + 1e-8)


# =================== 可视化类型 ===================

def viz_type1_difference_maps(ct, deformed, registered, gt, flow, save_prefix, sample_id):
    """
    类型1: 差异热力图对比
    放大显示配准前后与GT的差异
    """
    ct_np = to_np(ct)
    def_np = to_np(deformed)
    reg_np = to_np(registered)
    gt_np = to_np(gt)
    
    # 差异图
    diff_before = np.abs(def_np - gt_np)
    diff_after = np.abs(reg_np - gt_np)
    diff_improvement = diff_before - diff_after  # 正值 = 改善
    
    # 放大差异以增强可视性
    amplify = 3.0
    diff_before_amp = np.clip(diff_before * amplify, 0, 1)
    diff_after_amp = np.clip(diff_after * amplify, 0, 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 第一行: 原始图像
    axes[0, 0].imshow(ct_np, cmap='gray')
    axes[0, 0].set_title('(a) 固定图像 (CT)', fontsize=11)
    axes[0, 1].imshow(def_np, cmap='gray')
    axes[0, 1].set_title('(b) 形变MRI (配准前)', fontsize=11)
    axes[0, 2].imshow(reg_np, cmap='gray')
    axes[0, 2].set_title('(c) 配准结果', fontsize=11)
    axes[0, 3].imshow(gt_np, cmap='gray')
    axes[0, 3].set_title('(d) 真值MRI', fontsize=11)
    
    # 第二行: 差异图
    im1 = axes[1, 0].imshow(diff_before_amp, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('(e) 配准前误差 (×3放大)', fontsize=11)
    
    im2 = axes[1, 1].imshow(diff_after_amp, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('(f) 配准后误差 (×3放大)', fontsize=11)
    
    # 改善图 (蓝=退化, 红=改善)
    im3 = axes[1, 2].imshow(diff_improvement, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[1, 2].set_title('(g) 误差改善图 (红=改善)', fontsize=11)
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    # 变形场幅值
    flow_mag = compute_flow_magnitude(flow)
    im4 = axes[1, 3].imshow(flow_mag, cmap='jet')
    axes[1, 3].set_title('(h) 变形场幅值', fontsize=11)
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
    
    for ax in axes.flat:
        ax.axis('off')
    
    # 添加指标
    dice_before = compute_dice(deformed, gt)
    dice_after = compute_dice(registered, gt)
    mse_before = np.mean(diff_before ** 2)
    mse_after = np.mean(diff_after ** 2)
    
    fig.suptitle(
        f'样本 {sample_id}  |  DICE: {dice_before:.4f} → {dice_after:.4f} '
        f'(Δ={dice_after - dice_before:+.4f})  |  '
        f'MSE: {mse_before:.4f} → {mse_after:.4f} '
        f'(↓{(1 - mse_after / mse_before) * 100:.1f}%)',
        fontsize=12, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_prefix, f'sample{sample_id}_difference_maps.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()


def viz_type2_checkerboard(ct, deformed, registered, gt, flow, save_prefix, sample_id):
    """
    类型2: 棋盘格叠加对比
    经典配准可视化方法
    """
    def_np = to_np(deformed)
    reg_np = to_np(registered)
    gt_np = to_np(gt)
    
    checker_before = make_checkerboard(def_np, gt_np, n_tiles=8)
    checker_after = make_checkerboard(reg_np, gt_np, n_tiles=8)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(def_np, cmap='gray')
    axes[0].set_title('(a) 形变MRI', fontsize=11)
    
    axes[1].imshow(checker_before, cmap='gray')
    axes[1].set_title('(b) 棋盘格: 形变MRI vs 真值', fontsize=11)
    
    axes[2].imshow(checker_after, cmap='gray')
    axes[2].set_title('(c) 棋盘格: 配准结果 vs 真值', fontsize=11)
    
    axes[3].imshow(reg_np, cmap='gray')
    axes[3].set_title('(d) 配准结果', fontsize=11)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_prefix, f'sample{sample_id}_checkerboard.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()


def viz_type3_edge_overlay(ct, deformed, registered, gt, flow, save_prefix, sample_id):
    """
    类型3: 边缘叠加对比
    将真值MRI的边缘叠加在配准前后图上，显示对齐程度
    """
    def_np = to_np(deformed)
    reg_np = to_np(registered)
    gt_np = to_np(gt)
    
    overlay_before = make_edge_overlay(def_np, gt_np, color=(1.0, 0.2, 0.2))
    overlay_after = make_edge_overlay(reg_np, gt_np, color=(0.2, 1.0, 0.2))
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    
    axes[0].imshow(overlay_before)
    axes[0].set_title('(a) 配准前 + 真值边缘(红)', fontsize=11)
    
    axes[1].imshow(overlay_after)
    axes[1].set_title('(b) 配准后 + 真值边缘(绿)', fontsize=11)
    
    # 混合叠加
    gt_uint8 = (gt_np * 255).astype(np.uint8)
    edges = cv2.Canny(gt_uint8, 30, 100).astype(np.float32) / 255.0
    blend = np.stack([def_np, reg_np, gt_np], axis=-1)  # R=before, G=after, B=GT
    for c in range(3):
        blend[:, :, c] = blend[:, :, c] * (1 - edges * 0.5) + edges * 0.5
    axes[2].imshow(blend)
    axes[2].set_title('(c) RGB合成 (R=配准前, G=配准后, B=真值)', fontsize=11)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_prefix, f'sample{sample_id}_edge_overlay.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()


def viz_type4_deformation_field(ct, deformed, registered, gt, flow, save_prefix, sample_id):
    """
    类型4: 变形场网格可视化 + 幅值/方向
    """
    ct_np = to_np(ct)
    def_np = to_np(deformed)
    reg_np = to_np(registered)
    
    grid = make_deformation_grid(flow, spacing=6)
    flow_mag = compute_flow_magnitude(flow)
    
    # 变形场方向（HSV编码）
    flow_np = flow.detach().cpu().numpy()
    if flow_np.ndim == 4:
        flow_np = flow_np[0]
    angle = np.arctan2(flow_np[1], flow_np[0])  # [-pi, pi]
    mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    mag_norm = mag / (mag.max() + 1e-8)
    
    hsv = np.zeros((*angle.shape, 3), dtype=np.float32)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi)  # Hue [0,1]
    hsv[:, :, 1] = 1.0
    hsv[:, :, 2] = mag_norm
    
    from matplotlib.colors import hsv_to_rgb
    flow_color = hsv_to_rgb(hsv)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(ct_np, cmap='gray')
    axes[0].set_title('(a) 固定图像 (CT)', fontsize=11)
    
    axes[1].imshow(grid)
    axes[1].set_title('(b) 变形网格', fontsize=11)
    
    im = axes[2].imshow(flow_mag, cmap='jet')
    axes[2].set_title('(c) 变形幅值', fontsize=11)
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    axes[3].imshow(flow_color)
    axes[3].set_title('(d) 变形方向 (HSV编码)', fontsize=11)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_prefix, f'sample{sample_id}_deformation_field.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()


def viz_type5_ssim_comparison(ct, deformed, registered, gt, flow, save_prefix, sample_id):
    """
    类型5: 局部SSIM对比图
    """
    def_np = to_np(deformed)
    reg_np = to_np(registered)
    gt_np = to_np(gt)
    
    ssim_before, ssim_map_before = compute_local_ssim_map(def_np, gt_np, win_size=7)
    ssim_after, ssim_map_after = compute_local_ssim_map(reg_np, gt_np, win_size=7)
    ssim_improve = ssim_map_after - ssim_map_before
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    
    im1 = axes[0].imshow(ssim_map_before, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    axes[0].set_title(f'(a) SSIM图: 配准前 (均值={ssim_before:.4f})', fontsize=10)
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    im2 = axes[1].imshow(ssim_map_after, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    axes[1].set_title(f'(b) SSIM图: 配准后 (均值={ssim_after:.4f})', fontsize=10)
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    im3 = axes[2].imshow(ssim_improve, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    axes[2].set_title(f'(c) SSIM改善图 (Δ={ssim_after - ssim_before:+.4f})', fontsize=10)
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_prefix, f'sample{sample_id}_ssim_map.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()


def viz_type6_comprehensive(ct, deformed, registered, gt, flow, save_prefix, sample_id):
    """
    类型6: 综合全景对比图（放入论文的核心图）
    上行: CT | 形变MRI | 配准结果 | 真值MRI
    中行: 配准前差异 | 配准后差异 | 棋盘格(前) | 棋盘格(后)
    下行: 变形网格 | 变形幅值 | SSIM改善图 | 边缘叠加(后)
    """
    ct_np = to_np(ct)
    def_np = to_np(deformed)
    reg_np = to_np(registered)
    gt_np = to_np(gt)
    
    diff_before = np.abs(def_np - gt_np)
    diff_after = np.abs(reg_np - gt_np)
    diff_improve = diff_before - diff_after
    
    checker_before = make_checkerboard(def_np, gt_np)
    checker_after = make_checkerboard(reg_np, gt_np)
    
    grid = make_deformation_grid(flow, spacing=6)
    flow_mag = compute_flow_magnitude(flow)
    
    ssim_before, ssim_map_before = compute_local_ssim_map(def_np, gt_np, win_size=7)
    ssim_after, ssim_map_after = compute_local_ssim_map(reg_np, gt_np, win_size=7)
    ssim_improve = ssim_map_after - ssim_map_before
    
    edge_overlay = make_edge_overlay(reg_np, gt_np, color=(0.2, 1.0, 0.2))
    
    dice_before = compute_dice(deformed, gt)
    dice_after = compute_dice(registered, gt)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.25, wspace=0.15)
    
    # === 第一行: 原始图像 ===
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(ct_np, cmap='gray'); ax.set_title('(a) CT (固定图像)', fontsize=10); ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(def_np, cmap='gray'); ax.set_title('(b) 形变MRI (移动图像)', fontsize=10); ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(reg_np, cmap='gray'); ax.set_title('(c) 配准结果', fontsize=10); ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(gt_np, cmap='gray'); ax.set_title('(d) 真值MRI', fontsize=10); ax.axis('off')
    
    # === 第二行: 差异与棋盘格 ===
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(diff_before * 3, cmap='hot', vmin=0, vmax=1)
    ax.set_title('(e) 配准前误差 (×3)', fontsize=10); ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(diff_after * 3, cmap='hot', vmin=0, vmax=1)
    ax.set_title('(f) 配准后误差 (×3)', fontsize=10); ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(checker_before, cmap='gray')
    ax.set_title('(g) 棋盘格: 形变MRI vs 真值', fontsize=10); ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(checker_after, cmap='gray')
    ax.set_title('(h) 棋盘格: 配准后 vs 真值', fontsize=10); ax.axis('off')
    
    # === 第三行: 变形场与指标 ===
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(grid)
    ax.set_title('(i) 变形网格', fontsize=10); ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(flow_mag, cmap='jet')
    ax.set_title('(j) 变形幅值', fontsize=10); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(ssim_improve, cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    ax.set_title('(k) SSIM改善图', fontsize=10); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = fig.add_subplot(gs[2, 3])
    ax.imshow(edge_overlay)
    ax.set_title('(l) 配准后+真值边缘(绿)', fontsize=10); ax.axis('off')
    
    fig.suptitle(
        f'配准结果综合对比 | 样本 {sample_id}\n'
        f'DICE: {dice_before:.4f} → {dice_after:.4f} (Δ={dice_after-dice_before:+.4f})  |  '
        f'SSIM: {ssim_before:.4f} → {ssim_after:.4f} (Δ={ssim_after-ssim_before:+.4f})',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    plt.savefig(os.path.join(save_prefix, f'sample{sample_id}_comprehensive.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()


def viz_type7_before_after_summary(all_results, save_prefix):
    """
    类型7: 多样本汇总对比 (选取配准效果最好和中等的样本)
    """
    # 按DICE改善量排序
    sorted_results = sorted(all_results, key=lambda r: r['dice_improve'], reverse=True)
    
    # 选取改善最大的3个样本
    top_samples = sorted_results[:min(3, len(sorted_results))]
    
    n = len(top_samples)
    fig, axes = plt.subplots(n, 6, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    
    for row, r in enumerate(top_samples):
        def_np = to_np(r['deformed'])
        reg_np = to_np(r['registered'])
        gt_np = to_np(r['gt'])
        diff_before = np.abs(def_np - gt_np) * 3
        diff_after = np.abs(reg_np - gt_np) * 3
        checker = make_checkerboard(reg_np, gt_np)
        
        axes[row, 0].imshow(def_np, cmap='gray')
        axes[row, 0].set_title('形变MRI' if row == 0 else '', fontsize=10)
        axes[row, 0].set_ylabel(f"样本{r['id']}\nΔDICE={r['dice_improve']:+.4f}", fontsize=10, fontweight='bold')
        
        axes[row, 1].imshow(reg_np, cmap='gray')
        axes[row, 1].set_title('配准结果' if row == 0 else '', fontsize=10)
        
        axes[row, 2].imshow(gt_np, cmap='gray')
        axes[row, 2].set_title('真值MRI' if row == 0 else '', fontsize=10)
        
        axes[row, 3].imshow(np.clip(diff_before, 0, 1), cmap='hot', vmin=0, vmax=1)
        axes[row, 3].set_title('配准前误差(×3)' if row == 0 else '', fontsize=10)
        
        axes[row, 4].imshow(np.clip(diff_after, 0, 1), cmap='hot', vmin=0, vmax=1)
        axes[row, 4].set_title('配准后误差(×3)' if row == 0 else '', fontsize=10)
        
        axes[row, 5].imshow(checker, cmap='gray')
        axes[row, 5].set_title('棋盘格对比' if row == 0 else '', fontsize=10)
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([]) if ax.get_ylabel() == '' else None
    
    fig.suptitle('配准效果最佳样本汇总', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_prefix, 'multi_sample_summary.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()


# =================== 主流程 ===================

def main():
    print("=" * 60)
    print("增强配准结果可视化")
    print("=" * 60)
    
    model, device = load_model()
    test_loader = load_test_data(device)
    
    all_results = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            ct = batch['ct'].to(device)
            mri = batch['mri'].to(device)
            deformed = batch['deformed_mri'].to(device)
            
            outputs = model(ct, deformed)
            registered = outputs['warped_lvl0']
            flow = outputs['flow_lvl0']
            
            dice_before = compute_dice(deformed, mri)
            dice_after = compute_dice(registered, mri)
            
            result = {
                'id': i,
                'ct': ct, 'deformed': deformed,
                'registered': registered, 'gt': mri, 'flow': flow,
                'dice_before': dice_before, 'dice_after': dice_after,
                'dice_improve': dice_after - dice_before
            }
            all_results.append(result)
            
            print(f"样本 {i}: DICE {dice_before:.4f} → {dice_after:.4f} (Δ={dice_after-dice_before:+.4f})")
            
            # 为每个样本生成全部可视化（前5个样本）
            if i < 5:
                print(f"  生成样本 {i} 的增强可视化...")
                viz_type1_difference_maps(ct, deformed, registered, mri, flow, SAVE_DIR, i)
                viz_type2_checkerboard(ct, deformed, registered, mri, flow, SAVE_DIR, i)
                viz_type3_edge_overlay(ct, deformed, registered, mri, flow, SAVE_DIR, i)
                viz_type4_deformation_field(ct, deformed, registered, mri, flow, SAVE_DIR, i)
                viz_type5_ssim_comparison(ct, deformed, registered, mri, flow, SAVE_DIR, i)
                viz_type6_comprehensive(ct, deformed, registered, mri, flow, SAVE_DIR, i)
            
            # 只对综合图和差异图做全样本
            elif i < 10:
                viz_type1_difference_maps(ct, deformed, registered, mri, flow, SAVE_DIR, i)
                viz_type6_comprehensive(ct, deformed, registered, mri, flow, SAVE_DIR, i)
    
    # 生成多样本汇总图
    print("\n生成多样本汇总对比图...")
    viz_type7_before_after_summary(all_results, SAVE_DIR)
    
    # 打印统计
    dice_improvements = [r['dice_improve'] for r in all_results]
    print(f"\n{'='*60}")
    print(f"全部 {len(all_results)} 个样本统计:")
    print(f"  DICE改善: 均值={np.mean(dice_improvements):+.4f}, "
          f"中位数={np.median(dice_improvements):+.4f}")
    print(f"  改善样本比例: {sum(1 for d in dice_improvements if d > 0)}/{len(dice_improvements)}")
    print(f"  最大改善: {max(dice_improvements):+.4f}")
    print(f"  图片保存至: {SAVE_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
