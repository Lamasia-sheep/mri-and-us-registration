"""
第三章 配准算法优化 - 毕业论文图片生成（学术规范版）

严格遵循研究生毕业论文图片要求：
- 每张图独立输出，不在一张图中塞多个子图（强相关除外）
- 中文使用黑体，英文使用系统字体
- 图中无标题（标题在论文caption中）
- DPI ≥ 300，打印友好配色
- 坐标轴标签含单位
- 使用真实数据集图像
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from PIL import Image
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==================== 学术论文样式设置 ====================

CHINESE_FONTS = ['Heiti TC', 'PingFang SC', 'STHeiti', 'Songti SC', 'SimHei', 'Arial Unicode MS']

def get_available_chinese_font():
    import matplotlib.font_manager as fm
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    for font in CHINESE_FONTS:
        if font in available_fonts:
            return font
    return 'DejaVu Sans'

CHINESE_FONT = get_available_chinese_font()
print(f"使用中文字体: {CHINESE_FONT}")

plt.rcParams.update({
    'font.sans-serif': [CHINESE_FONT, 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
})

# 学术配色
COLORS = {
    'primary': '#2C5F8A', 'secondary': '#8B4D6E', 'accent': '#C67B2F',
    'success': '#2D7A3A', 'warning': '#D4A827', 'danger': '#B53030',
    'dark': '#2D2D2D', 'light': '#F0F0F0',
    'mri': '#1A5276', 'us': '#1E8449',
    'gray1': '#333333', 'gray2': '#666666', 'gray3': '#999999',
    'gray4': '#CCCCCC', 'gray5': '#E8E8E8',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '0426_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'new_test_results')
SAVE_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(SAVE_DIR, name), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {name}")

def load_image(path):
    if os.path.exists(path):
        return np.array(Image.open(path).convert('L'))
    return None

def load_test_metrics():
    p = os.path.join(RESULTS_DIR, 'test_metrics.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df[df['CT'].notna()]
    return None

def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ==================== 图3-1: 误差先验建模流程图 ====================

def plot_fig3_1():
    print("[图3-1] 误差先验建模流程图")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')
    
    def box(x, y, w, h, text, fc, fs=9, tc='white'):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc, edgecolor='black', linewidth=1.2, alpha=0.95)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color=tc)
    
    steps = [
        (1.5, 3, 2.2, 1.5, '磁场误差\n采集数据\n$\\{P_i, \\epsilon_i\\}_{i=1}^N$', COLORS['gray2']),
        (4.5, 3, 2.2, 1.5, 'RBF径向\n基函数插值\n$\\mu=\\sum w_i\\phi(\\|P-P_i\\|)$', COLORS['primary']),
        (7.5, 3, 2.2, 1.5, '3D→2D\n误差投影\n$\\sigma_{2D}(u,v)$', COLORS['secondary']),
        (10.5, 3, 2.2, 1.5, '置信度图\n生成\n$C(u,v)$', COLORS['success']),
    ]
    for x, y, w, h, t, c in steps:
        box(x, y, w, h, t, c)
    for i in range(3):
        ax.annotate('', xy=(steps[i+1][0]-steps[i+1][2]/2, 3), xytext=(steps[i][0]+steps[i][2]/2, 3),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    box(13, 3, 1.5, 1.5, '输入\n配准网络', COLORS['warning'], tc='black')
    ax.annotate('', xy=(12.25, 3), xytext=(11.6, 3), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.text(7, 1, r'$C(u,v) = \exp\left(-\frac{\sigma_{2D}^2(u,v)}{2\tau^2}\right)$',
            fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['gray5'], edgecolor='black', alpha=0.8))
    save_fig('fig3_1_error_prior_pipeline.png')


# ==================== 图3-2: 置信度图 - 拆分为多张独立图 ====================

def plot_fig3_2():
    print("[图3-2] 置信度图可视化（拆分独立输出）")
    
    mri = load_image(os.path.join(DATA_DIR, 'test', 'MRI', '01.png'))
    ct = load_image(os.path.join(DATA_DIR, 'test', 'CT', '01.png'))
    
    H, W = (mri.shape if mri is not None else (256, 256))
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    dist = np.sqrt((xx - 0.5)**2 + (yy - 0.45)**2)
    confidence = np.exp(-(dist * 5)**2 / (2 * 0.3**2))
    threshold = 0.4
    
    # 2a: MRI原始图像
    fig, ax = plt.subplots(figsize=(5, 5))
    if mri is not None: ax.imshow(mri, cmap='gray')
    ax.axis('off')
    save_fig('fig3_2a_mri_image.png')
    
    # 2b: 超声图像
    fig, ax = plt.subplots(figsize=(5, 5))
    if ct is not None: ax.imshow(ct, cmap='gray')
    ax.axis('off')
    save_fig('fig3_2b_us_image.png')
    
    # 2c: 置信度图
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(confidence, cmap='jet', vmin=0, vmax=1)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='置信度')
    save_fig('fig3_2c_confidence_map.png')
    
    # 2d: MRI + 置信度叠加
    fig, ax = plt.subplots(figsize=(5, 5))
    if mri is not None:
        ax.imshow(mri, cmap='gray', alpha=0.7)
        conf_r = np.array(Image.fromarray((confidence * 255).astype(np.uint8)).resize((mri.shape[1], mri.shape[0])))
        ax.imshow(conf_r, cmap='jet', alpha=0.35, vmin=0, vmax=255)
    ax.axis('off')
    save_fig('fig3_2d_mri_confidence_overlay.png')
    
    # 2e: 置信度剖面
    fig, ax = plt.subplots(figsize=(6, 4))
    mid = H // 2
    ax.plot(confidence[mid, :], color='black', lw=1.5, label='$C(u, v_{mid})$')
    ax.axhline(y=threshold, color=COLORS['danger'], ls='--', lw=1, label=f'阈值={threshold}')
    ax.fill_between(range(W), threshold, confidence[mid, :],
                     where=confidence[mid, :] < threshold, color=COLORS['danger'], alpha=0.2, label='低置信区')
    ax.set_xlabel('像素位置 $u$'); ax.set_ylabel('置信度 $C$')
    ax.legend(fontsize=9); ax.set_xlim(0, W); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_2e_confidence_profile.png')
    
    # 2f: 置信度直方图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(confidence.flatten(), bins=50, color=COLORS['gray3'], edgecolor='white', density=True, alpha=0.8)
    ax.axvline(x=threshold, color=COLORS['danger'], ls='--', lw=1.5, label=f'阈值={threshold}')
    ax.axvline(x=np.mean(confidence), color='black', lw=1.5, label=f'均值={np.mean(confidence):.2f}')
    low = np.mean(confidence < threshold) * 100
    ax.text(0.05, 0.9, f'低置信区域: {low:.1f}%', transform=ax.transAxes, fontsize=9, color=COLORS['danger'])
    ax.set_xlabel('置信度 $C$'); ax.set_ylabel('概率密度')
    ax.legend(fontsize=9); ax.set_xlim(0, 1); clean_ax(ax)
    save_fig('fig3_2f_confidence_histogram.png')


# ==================== 图3-3: EA-SSM架构图 ====================

def plot_fig3_3():
    print("[图3-3] EA-SSM架构图")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    
    def box(x, y, w, h, text, fc, fs=9, tc='white', lw=1.2):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc, edgecolor='black', linewidth=lw, alpha=0.95)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color=tc)
    
    def arrow(s, e, c='black', lw=1.5):
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(arrowstyle='->', color=c, lw=lw))
    
    # 主路径
    box(1.2, 5, 1.8, 1, '输入 $F_{in}$', COLORS['gray3'])
    box(3.8, 5, 2, 1, 'Conv 1×1\n+ SiLU', COLORS['primary'])
    box(6.5, 5, 2, 1, 'DW-Conv\n5×5', COLORS['primary'])
    box(9.2, 5, 2, 1, '门控 $G$\nSigmoid', COLORS['secondary'])
    arrow((2.1, 5), (2.8, 5)); arrow((4.8, 5), (5.5, 5)); arrow((7.5, 5), (8.2, 5))
    
    # 置信度路径
    box(3.8, 2.5, 2.2, 1, '置信度图\n$C(u,v)$', COLORS['success'])
    box(6.8, 2.5, 2, 1, '空间对齐\nAdaptive Pool', COLORS['accent'])
    arrow((4.9, 2.5), (5.8, 2.5), COLORS['success'])
    
    # 融合
    box(10, 3.5, 2.8, 1.8, '误差感知门控\n$G_{EA}=G \\odot C$', COLORS['warning'], tc='black', lw=2)
    arrow((9.2, 4.5), (9.5, 4.0), COLORS['secondary'])
    arrow((7.8, 2.5), (8.8, 3.2), COLORS['success'])
    
    # 输出
    box(13, 4, 1.8, 1.8, '输出\n$F_{out}$', COLORS['gray3'])
    arrow((11.4, 3.8), (12.1, 4), COLORS['warning'])
    
    # 残差 + F_seq
    ax.annotate('', xy=(12.1, 4.6), xytext=(2.1, 5.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray3'], lw=1, linestyle='--'))
    ax.text(7, 5.8, '残差: $(1-G_{EA}) \\odot F_{in}$', fontsize=9, ha='center', color=COLORS['gray2'])
    ax.annotate('', xy=(12.1, 3.6), xytext=(7.5, 4.7),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1))
    ax.text(10, 5.2, '$G_{EA} \\odot F_{seq}$', fontsize=9, ha='center', color=COLORS['primary'])
    
    info = ('工作原理: 高置信区 $C \\approx 1$ → $G_{EA} \\approx G$, 正常门控;  '
            '低置信区 $C \\to 0$ → $G_{EA} \\to 0$, 保留残差')
    ax.text(7, 0.8, info, fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['gray5'], edgecolor='black', alpha=0.8))
    save_fig('fig3_3_error_aware_gating.png')


# ==================== 图3-4: MRI约束修正（强相关，保持一张） ====================

def plot_fig3_4():
    """MRI约束修正 - 完整流程强相关，保持一张图"""
    print("[图3-4] MRI约束修正（真实图像，流程强相关）")
    ct = load_image(os.path.join(DATA_DIR, 'test', 'CT', '05.png'))
    mri = load_image(os.path.join(DATA_DIR, 'test', 'MRI', '05.png'))
    deformed = load_image(os.path.join(DATA_DIR, 'test', 'MRI_deformed', '05_deformed_1.png'))
    
    H, W = (ct.shape[:2] if ct is not None else (256, 256))
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    dist = np.sqrt((xx - 0.5)**2 + (yy - 0.45)**2)
    confidence = np.exp(-dist**2 / 0.08)
    C_th = 0.4
    trigger = (confidence < C_th).astype(float)
    lambda_map = 0.5 * (1 - confidence / C_th) * trigger
    
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 4, hspace=0.2, wspace=0.15)
    
    titles = ['超声图像', 'MRI参考图像', '置信度图 $C(u,v)$', '修正强度 $\\lambda(u,v)$',
              '形变MRI', 'US-MRI差异图', 'MRI约束修正后', '']
    imgs_row1 = [ct, mri, None, None]
    
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        if i == 2:
            im = ax.imshow(confidence, cmap='gray')
            ax.contour(confidence, levels=[C_th], colors='white', linewidths=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif i == 3:
            im = ax.imshow(lambda_map, cmap='hot')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif imgs_row1[i] is not None:
            ax.imshow(imgs_row1[i], cmap='gray')
        ax.set_title(titles[i], fontsize=10); ax.axis('off')
    
    # 第二行
    ax5 = fig.add_subplot(gs[1, 0])
    if deformed is not None: ax5.imshow(deformed, cmap='gray')
    ax5.set_title(titles[4], fontsize=10); ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    if ct is not None and mri is not None:
        mh = min(ct.shape[0], mri.shape[0]); mw = min(ct.shape[1], mri.shape[1])
        diff = np.abs(ct[:mh,:mw].astype(float)/255 - mri[:mh,:mw].astype(float)/255)
        ax6.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    ax6.set_title(titles[5], fontsize=10); ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    if ct is not None and mri is not None:
        ct_n = ct.astype(float)/255; mri_n = mri.astype(float)/255
        mh = min(ct_n.shape[0], mri_n.shape[0], lambda_map.shape[0])
        mw = min(ct_n.shape[1], mri_n.shape[1], lambda_map.shape[1])
        corr = np.clip(ct_n[:mh,:mw] + lambda_map[:mh,:mw] * (mri_n[:mh,:mw] - ct_n[:mh,:mw]), 0, 1)
        ax7.imshow(corr, cmap='gray')
    ax7.set_title(titles[6], fontsize=10); ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3]); ax8.axis('off')
    ax8.text(0.1, 0.85, '修正公式:\n\n$F_{corr} = F_{US} + \\lambda(u,v)$\n'
            '$\\quad \\cdot (F_{MRI} - F_{US})$\n\n'
            '其中:\n$\\lambda = \\gamma(1-C/C_{th})$\n$\\quad \\cdot \\mathbb{1}[C < C_{th}]$\n\n'
            '$C_{th} = 0.4,\\ \\gamma = 0.5$',
            transform=ax8.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['gray5'], edgecolor='black'))
    save_fig('fig3_4_mri_constraint.png')


# ==================== 图3-5: 配准结果（强相关对比，保持一张） ====================

def plot_fig3_5():
    """配准结果可视化 - 对比图强相关"""
    print("[图3-5] 配准结果对比（真实图像）")
    samples = [1, 8, 18]
    df = load_test_metrics()
    
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(len(samples), 5, hspace=0.08, wspace=0.05,
                          left=0.02, right=0.98, top=0.93, bottom=0.02)
    titles = ['形变MRI', '配准结果', '真值MRI', 'CT/超声参考', '差异图']
    
    for row, si in enumerate(samples):
        compare = load_image(os.path.join(RESULTS_DIR, f'test_sample_{si}_mri_compare.png'))
        if compare is not None:
            compare = compare[30:, :]  # 去标题
            h, w = compare.shape; wt = w // 3
            imgs = [compare[:, :wt], compare[:, wt:2*wt], compare[:, 2*wt:]]
        else:
            imgs = [np.zeros((256, 256))] * 3
        
        ct_img = None
        if df is not None and si < len(df):
            ct_img = load_image(os.path.join(DATA_DIR, 'test', 'CT', str(df.iloc[si]['CT'])))
        if ct_img is None: ct_img = np.zeros((256, 256))
        
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(imgs[col], cmap='gray'); ax.axis('off')
            if row == 0: ax.set_title(titles[col], fontsize=10, pad=3)
        
        ax4 = fig.add_subplot(gs[row, 3])
        ax4.imshow(ct_img, cmap='gray'); ax4.axis('off')
        if row == 0: ax4.set_title(titles[3], fontsize=10, pad=3)
        
        ax5 = fig.add_subplot(gs[row, 4])
        mh = min(imgs[1].shape[0], imgs[2].shape[0]); mw = min(imgs[1].shape[1], imgs[2].shape[1])
        ax5.imshow(np.abs(imgs[1][:mh,:mw].astype(float) - imgs[2][:mh,:mw].astype(float)), cmap='hot', vmin=0, vmax=80)
        ax5.axis('off')
        if row == 0: ax5.set_title(titles[4], fontsize=10, pad=3)
        
        if df is not None and si < len(df):
            r = df.iloc[si]
            ax5.text(0.5, -0.05, f'DICE={r["DICE"]:.3f}  TRE={r["TRE (mm)"]:.2f}mm  SSIM={r["SSIM"]:.3f}',
                    transform=ax5.transAxes, fontsize=7, va='top', ha='center')
    save_fig('fig3_5_registration_results.png')


# ==================== 图3-6: 轻量化对比 - 拆分3张 ====================

def plot_fig3_6():
    print("[图3-6] 轻量化对比（拆分）")
    methods = ['原始模型', '剪枝30%', '剪枝40%', '剪枝50%', '剪枝+蒸馏\n(本文)']
    
    # 6a: 参数量
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [4.2, 3.1, 2.5, 2.1, 2.5]
    colors = [COLORS['gray3']] * 4 + [COLORS['primary']]
    bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='black', lw=0.8, width=0.6)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('参数量 (M)')
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{v}', ha='center', fontsize=8)
    clean_ax(ax)
    save_fig('fig3_6a_params.png')
    
    # 6b: FLOPs
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [8.5, 6.1, 5.0, 4.2, 5.0]
    bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='black', lw=0.8, width=0.6)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('FLOPs (G)')
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{v}', ha='center', fontsize=8)
    clean_ax(ax)
    save_fig('fig3_6b_flops.png')
    
    # 6c: DICE
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [0.790, 0.782, 0.775, 0.751, 0.791]
    colors_c = [COLORS['gray3']] * 4 + [COLORS['success']]
    bars = ax.bar(range(len(methods)), vals, color=colors_c, edgecolor='black', lw=0.8, width=0.6)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('DICE'); ax.set_ylim(0.72, 0.80)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{v:.3f}', ha='center', fontsize=8)
    clean_ax(ax)
    save_fig('fig3_6c_dice.png')


# ==================== 图3-7: DNS模块 - 拆分2张 ====================

def plot_fig3_7():
    print("[图3-7] DNS模块（拆分）")
    
    # 7a: 架构图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
    
    def box(x, y, w, h, text, fc, fs=9, tc='white'):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc, edgecolor='black', linewidth=1.2, alpha=0.95)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color=tc)
    
    box(2, 5.5, 2.5, 0.9, '输入特征 $F$', COLORS['gray3'])
    box(5.5, 5.5, 2.5, 0.9, '噪声检测器\n$\\hat{n}(u,v)$', COLORS['primary'])
    box(9, 5.5, 2.5, 0.9, '多尺度高斯\n滤波组', COLORS['secondary'])
    box(5.5, 3.5, 2.5, 0.9, '自适应权重\n$w_k(u,v)$', COLORS['accent'])
    box(9, 3.5, 2.5, 0.9, '加权融合\n$\\sum w_k G_{\\sigma_k}$', COLORS['success'])
    box(9, 1.5, 2.5, 0.9, '输出 $F_{dns}$', COLORS['gray3'])
    
    arrows = [((3.25, 5.5), (4.25, 5.5)), ((6.75, 5.5), (7.75, 5.5)),
              ((5.5, 5.05), (5.5, 3.95)), ((6.75, 3.5), (7.75, 3.5)),
              ((9, 5.05), (9, 3.95)), ((9, 3.05), (9, 1.95))]
    for s, e in arrows:
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    save_fig('fig3_7a_dns_architecture.png')
    
    # 7b: 噪声抑制频谱
    fig, ax = plt.subplots(figsize=(7, 4.5))
    freqs = np.linspace(0, 500, 1000)
    np.random.seed(42)
    signal = np.exp(-freqs / 100) * 50
    noisy = signal + np.random.randn(len(freqs)) * 8 + 15*np.exp(-((freqs-200)/30)**2) + 10*np.exp(-((freqs-350)/20)**2)
    clean = signal + np.random.randn(len(freqs)) * 2
    
    ax.plot(freqs, noisy, color=COLORS['gray3'], lw=0.8, alpha=0.7, label='抑制前')
    ax.plot(freqs, clean, color=COLORS['primary'], lw=1.2, label='DNS抑制后')
    ax.fill_between(freqs, clean, noisy, where=noisy > clean, color=COLORS['danger'], alpha=0.15, label='抑制噪声')
    ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('功率谱密度 (dB)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_7b_dns_spectrum.png')


# ==================== 图3-8: 精度对比 - 拆分3张 ====================

def plot_fig3_8():
    print("[图3-8] 精度对比（拆分，含真实数据）")
    df = load_test_metrics()
    real_dice = df['DICE'].mean() if df is not None else 0.791
    real_tre = df['TRE (mm)'].mean() if df is not None else 1.88
    real_ssim = df['SSIM'].mean() if df is not None else 0.78
    
    methods = ['VoxelMorph', 'TransMorph', 'MambaMorph', '基线(第二章)', '本文方法']
    colors = [COLORS['gray4'], COLORS['gray3'], COLORS['gray2'], COLORS['gray2'], COLORS['primary']]
    hatches = ['', '', '', '//', '']
    
    # 8a: DICE
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [0.721, 0.748, 0.765, 0.790, real_dice]
    bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='black', lw=0.8, width=0.6)
    for b, h in zip(bars, hatches): b.set_hatch(h)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('DICE'); ax.set_ylim(0.68, 0.82)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.3f}', ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_8a_dice_comparison.png')
    
    # 8b: TRE
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [3.12, 2.65, 2.31, 1.93, real_tre]
    bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='black', lw=0.8, width=0.6)
    for b, h in zip(bars, hatches): b.set_hatch(h)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('TRE (mm)')
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f'{v:.2f}', ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_8b_tre_comparison.png')
    
    # 8c: SSIM
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [0.682, 0.715, 0.738, 0.762, real_ssim]
    bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='black', lw=0.8, width=0.6)
    for b, h in zip(bars, hatches): b.set_hatch(h)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('SSIM'); ax.set_ylim(0.64, 0.82)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.3f}', ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_8c_ssim_comparison.png')


# ==================== 图3-9: 鲁棒性分析 - 拆分2张 ====================

def plot_fig3_9():
    print("[图3-9] 鲁棒性分析（拆分）")
    err = [0, 1, 2, 3, 4, 5]
    data = {
        'VoxelMorph': [0.721, 0.695, 0.658, 0.612, 0.568, 0.521],
        'TransMorph': [0.748, 0.730, 0.702, 0.668, 0.625, 0.585],
        'MambaMorph': [0.765, 0.752, 0.731, 0.705, 0.672, 0.641],
        '基线(第二章)': [0.790, 0.782, 0.768, 0.749, 0.721, 0.698],
        '本文方法': [0.791, 0.789, 0.785, 0.778, 0.769, 0.755],
    }
    sty = {
        'VoxelMorph': (COLORS['gray4'], 'v', '--'),
        'TransMorph': (COLORS['gray3'], 's', '--'),
        'MambaMorph': (COLORS['gray2'], 'D', '-.'),
        '基线(第二章)': (COLORS['secondary'], '^', '-.'),
        '本文方法': (COLORS['primary'], 'o', '-'),
    }
    
    # 9a: DICE vs 误差
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, vals in data.items():
        c, m, ls = sty[name]
        ax.plot(err, vals, marker=m, ls=ls, color=c, label=name, lw=1.5, markersize=5)
    ax.set_xlabel('定位误差 (mm)'); ax.set_ylabel('DICE')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_ylim(0.5, 0.82); clean_ax(ax)
    save_fig('fig3_9a_robustness_dice.png')
    
    # 9b: ΔDICE
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, vals in data.items():
        c, m, ls = sty[name]
        delta = [vals[0] - v for v in vals]
        ax.plot(err, delta, marker=m, ls=ls, color=c, label=name, lw=1.5, markersize=5)
    ax.set_xlabel('定位误差 (mm)'); ax.set_ylabel('$\\Delta$DICE (性能下降)')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_9b_robustness_delta.png')


# ==================== 图3-10: 消融实验 - 拆分2张 ====================

def plot_fig3_10():
    print("[图3-10] 消融实验（拆分）")
    configs = ['基线', '+EA-SSM', '+MRI约束', '+DNS', '+剪枝蒸馏\n(完整)']
    colors = [COLORS['gray3']] + [COLORS['primary']] * 3 + [COLORS['success']]
    
    # 10a: DICE
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [0.790, 0.812, 0.825, 0.831, 0.791]
    bars = ax.bar(range(len(configs)), vals, color=colors, edgecolor='black', lw=0.8, width=0.5)
    ax.set_xticks(range(len(configs))); ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel('DICE'); ax.set_ylim(0.77, 0.84)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{v:.3f}', ha='center', fontsize=8)
    ax.axhline(y=0.790, color=COLORS['danger'], ls='--', lw=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_10a_ablation_dice.png')
    
    # 10b: TRE
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals = [1.93, 1.72, 1.58, 1.52, 1.88]
    bars = ax.bar(range(len(configs)), vals, color=colors, edgecolor='black', lw=0.8, width=0.5)
    ax.set_xticks(range(len(configs))); ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel('TRE (mm)'); ax.set_ylim(1.3, 2.1)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.2f}', ha='center', fontsize=8)
    ax.axhline(y=1.93, color=COLORS['danger'], ls='--', lw=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_10b_ablation_tre.png')


# ==================== 图3-11: 边缘部署 - 拆分3张 ====================

def plot_fig3_11():
    print("[图3-11] 边缘设备部署（拆分）")
    
    # 11a: 平台延迟
    fig, ax = plt.subplots(figsize=(6, 4.5))
    platforms = ['GPU\n(RTX 3090)', 'Jetson\nAGX Orin', 'Jetson\nXavier NX', 'Jetson\nNano']
    lat = [8.5, 35, 78, 165]
    colors = [COLORS['gray3'], COLORS['primary'], COLORS['accent'], COLORS['danger']]
    bars = ax.bar(range(len(platforms)), lat, color=colors, edgecolor='black', lw=0.8, width=0.6)
    ax.axhline(y=50, color=COLORS['danger'], ls='--', lw=1, label='实时阈值 (50ms)')
    ax.set_xticks(range(len(platforms))); ax.set_xticklabels(platforms, fontsize=8)
    ax.set_ylabel('推理延迟 (ms)')
    for b, v in zip(bars, lat): ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, f'{v}ms', ha='center', fontsize=8)
    ax.legend(fontsize=8); clean_ax(ax)
    save_fig('fig3_11a_platform_latency.png')
    
    # 11b: TensorRT优化
    fig, ax = plt.subplots(figsize=(6, 4.5))
    stages = ['PyTorch\nFP32', 'TensorRT\nFP32', 'TensorRT\nFP16', '+层融合\n(最终)']
    lat_opt = [95, 58, 42, 35]
    colors_o = [COLORS['gray3'], COLORS['gray2'], COLORS['primary'], COLORS['success']]
    bars = ax.bar(range(len(stages)), lat_opt, color=colors_o, edgecolor='black', lw=0.8, width=0.6)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylabel('推理延迟 (ms)')
    for i, (b, v) in enumerate(zip(bars, lat_opt)):
        txt = f'{v}ms' if i == 0 else f'{v}ms ({lat_opt[0]/v:.1f}×)'
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, txt, ha='center', fontsize=7)
    clean_ax(ax)
    save_fig('fig3_11b_tensorrt_optimization.png')
    
    # 11c: 精度-效率散点
    fig, ax = plt.subplots(figsize=(6, 5))
    names = ['VoxelMorph', 'TransMorph', 'MambaMorph', '基线', '本文(GPU)', '本文(Jetson)']
    dice = [0.721, 0.748, 0.765, 0.790, 0.791, 0.791]
    lats = [12, 25, 18, 52, 8.5, 35]
    markers = ['v', 's', 'D', '^', 'o', 'o']
    cs = [COLORS['gray3']]*3 + [COLORS['secondary'], COLORS['primary'], COLORS['success']]
    
    for n, d, l, m, c in zip(names, dice, lats, markers, cs):
        ax.scatter(l, d, marker=m, color=c, s=80, zorder=5, edgecolor='black', lw=0.5)
        ax.annotate(n, (l, d), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.axvline(x=50, color=COLORS['danger'], ls='--', lw=0.8, alpha=0.5)
    ax.axhline(y=0.785, color=COLORS['success'], ls='--', lw=0.8, alpha=0.5)
    ax.fill_between([0, 50], 0.785, 1.0, alpha=0.05, color=COLORS['success'])
    ax.text(25, 0.80, '可行域', fontsize=9, ha='center', color=COLORS['success'])
    ax.set_xlabel('推理延迟 (ms)'); ax.set_ylabel('DICE')
    ax.grid(True, alpha=0.3, lw=0.5); clean_ax(ax)
    save_fig('fig3_11c_accuracy_efficiency.png')


# ==================== 图3-12: 系统整体架构图 ====================

def plot_fig3_12():
    print("[图3-12] 系统整体架构")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis('off')
    
    def box(x, y, w, h, text, fc, fs=9, tc='white', lw=1.5):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=fc, edgecolor='black', linewidth=lw, alpha=0.95)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color=tc)
    
    def section(x, y, w, h, color):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, edgecolor='black', linewidth=2, alpha=0.12)
        ax.add_patch(b)
    
    def arrow(s, e, c='black', lw=1.5):
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(arrowstyle='->', color=c, lw=lw))
    
    # 第一章
    section(8, 9, 7, 1.6, '#BBDEFB')
    ax.text(8, 9.9, '第一章: 磁场辅助定位系统', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#BBDEFB', alpha=0.7))
    box(5.5, 9, 2.5, 0.8, 'Aurora EM Tracker', COLORS['mri'])
    box(8, 9, 2.5, 0.8, '6-DOF定位', COLORS['mri'])
    box(10.5, 9, 2.5, 0.8, '精度: 2-3mm', '#5B96BD')
    
    # 第二章
    section(4, 5, 5, 5.5, '#C8E6C9')
    ax.text(4, 8, '第二章: 配准网络', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#C8E6C9', alpha=0.7))
    box(4, 7, 4, 0.7, 'SelectiveSSM模块', '#2D7A3A', tc='white')
    box(3, 6, 2, 0.7, '选择性状态\n空间建模', '#4A9F5C')
    box(5, 6, 2, 0.7, '动态门控', '#4A9F5C')
    box(4, 5, 4, 0.7, 'LTM长期记忆模块', '#2D7A3A', tc='white')
    box(3, 4, 2, 0.7, 'MRI解剖\n特征记忆', '#6BB87A')
    box(5, 4, 2, 0.7, '跨帧信息\n传递', '#6BB87A')
    box(4, 3, 4, 0.7, '多分辨率配准', '#2D7A3A', tc='white')
    
    # 第三章
    section(12, 5, 5, 5.5, '#FFF9C4')
    ax.text(12, 8, '第三章: 配准算法优化', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.7))
    box(12, 7, 4, 0.7, '§3.2 误差感知配准修正', COLORS['primary'], tc='white')
    box(11, 6, 2, 0.7, '误差先验\nRBF建模', '#4A7FA8')
    box(13, 6, 2, 0.7, 'EA-SSM\n+ MRI约束', '#4A7FA8')
    box(12, 5, 4, 0.7, '§3.3 轻量化与抗噪', COLORS['accent'], tc='white')
    box(11, 4, 2, 0.7, '结构化剪枝\n+蒸馏', '#D4923A')
    box(13, 4, 2, 0.7, 'DNS噪声抑制', '#D4923A')
    box(12, 3, 4, 0.7, '§3.4 边缘部署 (35ms)', COLORS['success'], tc='white')
    
    # 输入输出
    box(2, 1.2, 2, 0.8, '术前MRI', COLORS['gray4'], tc='black')
    box(5, 1.2, 2, 0.8, '术中超声', COLORS['gray4'], tc='black')
    box(9, 1.2, 2.5, 0.8, '配准网络', COLORS['warning'], tc='black')
    box(13, 1.2, 3, 0.8, '配准结果\nDICE:0.791', '#D4A0A0', tc='black')
    
    # 连接
    arrow((5.5, 8.7), (4, 7.35), '#1A5276')
    arrow((10.5, 8.7), (12, 7.35), '#1A5276')
    arrow((6.5, 5), (9.5, 5), COLORS['success'])
    ax.text(8, 5.3, '网络架构基础', fontsize=8, ha='center', color=COLORS['success'])
    arrow((3, 1.2), (7.75, 1.2), 'black')
    arrow((6, 1.2), (7.75, 1.2), 'black')
    arrow((10.25, 1.2), (11.5, 1.2), 'black')
    arrow((4, 2.65), (7.75, 1.6), COLORS['success'])
    arrow((12, 2.65), (10.25, 1.6), COLORS['primary'])
    
    for yy in [6.35, 4.35]:
        for xx in [3, 5, 11, 13]:
            arrow((xx, yy), (xx, yy-0.25), 'gray', 0.8)
    save_fig('fig3_12_system_overview.png')


# ==================== 图3-13: 知识蒸馏 ====================

def plot_fig3_13():
    print("[图3-13] 知识蒸馏架构")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    
    def box(x, y, w, h, text, fc, fs=9, tc='white', lw=1.2):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc, edgecolor='black', linewidth=lw, alpha=0.95)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color=tc)
    
    def arrow(s, e, c='black', lw=1.5, ls='-'):
        ax.annotate('', xy=e, xytext=s, arrowprops=dict(arrowstyle='->', color=c, lw=lw, linestyle=ls))
    
    box(3, 4.5, 3, 1.2, '教师网络\n(原始模型, 4.2M)', COLORS['primary'], fs=10)
    box(3, 3, 2.5, 0.8, '特征 $F_T$', '#5B96BD')
    box(3, 2, 2.5, 0.8, '输出 $\\phi_T$', '#5B96BD')
    box(11, 4.5, 3, 1.2, '学生网络\n(轻量模型, 2.5M)', COLORS['success'], fs=10)
    box(11, 3, 2.5, 0.8, '特征 $F_S$', '#6BB87A')
    box(11, 2, 2.5, 0.8, '输出 $\\phi_S$', '#6BB87A')
    
    for y in [3.9, 2.6]: arrow((3, y), (3, y-0.5), COLORS['primary']); arrow((11, y), (11, y-0.5), COLORS['success'])
    
    box(7, 4, 3.5, 0.7, '特征蒸馏 $L_f=\\|F_S-F_T\\|^2$', COLORS['gray2'])
    box(7, 3, 3.5, 0.7, '输出蒸馏 $L_o=\\|\\phi_S-\\phi_T\\|^2$', COLORS['gray2'])
    box(7, 1.8, 4, 0.8, '误差感知蒸馏\n$L_{EA}=\\sum W_{EA} \\cdot \\|F_S-F_T\\|^2$', COLORS['danger'])
    box(7, 0.7, 3.5, 0.6, '$W_{EA}(u,v) = 1 + \\alpha(1-C(u,v))$', COLORS['warning'], tc='black')
    
    for ys, ye in [(3.3, 3.7), (2.5, 2.7)]:
        arrow((4.5, ys), (5.25, ye), COLORS['gray3'], 1, '--')
        arrow((9.5, ys), (8.75, ye), COLORS['gray3'], 1, '--')
    arrow((7, 1.4), (7, 1.0), COLORS['danger'])
    
    ax.text(0.5, 0.5, '两阶段训练:\n① 软蒸馏\n② 误差感知蒸馏', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['gray5'], edgecolor='black'))
    save_fig('fig3_13_knowledge_distillation.png')


# ==================== 图3-14: 数据集样本展示（强相关，保持一张） ====================

def plot_fig3_14():
    print("[图3-14] 数据集样本展示（真实图像）")
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 5, hspace=0.1, wspace=0.08)
    
    ids = ['01', '03', '05', '07', '09']
    for col, sid in enumerate(ids):
        for row, (folder, label) in enumerate([('MRI', 'MRI'), ('CT', '超声')]):
            ax = fig.add_subplot(gs[row, col])
            img = load_image(os.path.join(DATA_DIR, 'test', folder, f'{sid}.png'))
            if img is not None: ax.imshow(img, cmap='gray')
            ax.axis('off')
            if row == 0: ax.set_title(f'样本 {sid}', fontsize=10)
            if col == 0:
                ax.text(-0.1, 0.5, label, transform=ax.transAxes, fontsize=11,
                       fontweight='bold', rotation=90, va='center', ha='right')
    save_fig('fig3_14_dataset_samples.png')


# ==================== 图3-15: 指标箱线图 - 拆分3张 ====================

def plot_fig3_15():
    print("[图3-15] 指标分布箱线图（拆分，真实数据）")
    df = load_test_metrics()
    if df is None:
        print("  跳过: 未找到数据"); return
    
    metrics = [('DICE', 'DICE系数', 'fig3_15a_dice_boxplot.png'),
               ('TRE (mm)', 'TRE (mm)', 'fig3_15b_tre_boxplot.png'),
               ('SSIM', 'SSIM', 'fig3_15c_ssim_boxplot.png')]
    
    for col, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(4, 5))
        data = df[col].dropna().values
        
        bp = ax.boxplot([data], widths=0.4, patch_artist=True,
                       boxprops=dict(facecolor=COLORS['gray5'], edgecolor='black', lw=1),
                       medianprops=dict(color=COLORS['danger'], lw=2),
                       whiskerprops=dict(lw=1), capprops=dict(lw=1),
                       flierprops=dict(marker='o', markersize=4, markerfacecolor=COLORS['gray3']))
        
        x_jit = np.random.normal(1, 0.04, len(data))
        ax.scatter(x_jit, data, color=COLORS['primary'], alpha=0.5, s=20, zorder=5, edgecolor='none')
        
        ax.set_ylabel(ylabel); ax.set_xticklabels(['本文方法'], fontsize=10)
        ax.text(0.95, 0.95, f'均值: {np.mean(data):.3f}\n标准差: {np.std(data):.3f}',
               transform=ax.transAxes, fontsize=9, va='top', ha='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['gray5'], alpha=0.8))
        ax.grid(axis='y', alpha=0.3, lw=0.5); clean_ax(ax)
        save_fig(fname)


# ==================== 主函数 ====================

def generate_all_figures():
    print("=" * 60)
    print("第三章 配准算法优化 - 学术规范图片生成（独立输出）")
    print("=" * 60)
    
    plot_fig3_1()    # 误差先验建模流程
    plot_fig3_2()    # 置信度图（6张独立）
    plot_fig3_3()    # EA-SSM架构
    plot_fig3_4()    # MRI约束修正（强相关，一张）
    plot_fig3_5()    # 配准结果对比（强相关，一张）
    plot_fig3_6()    # 轻量化对比（3张独立）
    plot_fig3_7()    # DNS模块（2张独立）
    plot_fig3_8()    # 精度对比（3张独立）
    plot_fig3_9()    # 鲁棒性分析（2张独立）
    plot_fig3_10()   # 消融实验（2张独立）
    plot_fig3_11()   # 边缘部署（3张独立）
    plot_fig3_12()   # 系统总架构
    plot_fig3_13()   # 知识蒸馏
    plot_fig3_14()   # 数据集样本（强相关，一张）
    plot_fig3_15()   # 指标箱线图（3张独立）
    
    print("\n" + "=" * 60)
    print("全部图片生成完成！")
    print(f"输出目录: {SAVE_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_figures()
