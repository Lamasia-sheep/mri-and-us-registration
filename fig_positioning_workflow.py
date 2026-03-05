"""
磁场辅助定位结合超声探头 - 简洁工作流程图 (配实景图)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Arc, Wedge
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ---- 字体 ----
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
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
})

DATA_DIR = './0426_data/test'


# ---- 绘图工具 ----

def big_arrow(ax, x1, y1, x2, y2, color='#546E7A', lw=3.0, label=''):
    """粗箭头"""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 color=color, lw=lw, mutation_scale=20,
                                 shrinkA=8, shrinkB=8, zorder=5))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.35
        ax.text(mx, my, label, fontsize=11, ha='center', va='center',
                color=color, fontweight='bold', zorder=6)


def draw_rounded_box(ax, x, y, w, h, fc, ec, lw=1.5, zorder=3):
    """圆角矩形"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=lw, zorder=zorder))


def embed_image(ax, img_path, cx, cy, zoom=0.22):
    """嵌入真实图片"""
    img = Image.open(img_path).convert('L')
    img = img.resize((128, 128), Image.LANCZOS)
    img_arr = np.array(img)
    im = OffsetImage(img_arr, zoom=zoom, cmap='gray')
    ab = AnnotationBbox(im, (cx, cy), frameon=True,
                        bboxprops=dict(boxstyle='round,pad=0.05',
                                       edgecolor='#90A4AE', linewidth=1.2,
                                       facecolor='white'),
                        zorder=4)
    ax.add_artist(ab)


def draw_emitter(ax, cx, cy):
    """画磁场发射器示意图"""
    # 底座
    draw_rounded_box(ax, cx-0.8, cy-0.6, 1.6, 0.5, '#CFD8DC', '#607D8B', lw=1.2)
    # 立柱
    ax.add_patch(FancyBboxPatch((cx-0.15, cy-0.1), 0.3, 1.2, boxstyle="round,pad=0.02",
                                fc='#B0BEC5', ec='#607D8B', lw=1.0, zorder=3))
    # 发射头 (椭圆)
    ellipse = matplotlib.patches.Ellipse((cx, cy+1.3), 1.2, 0.6,
                                          fc='#78909C', ec='#455A64', lw=1.2, zorder=3)
    ax.add_patch(ellipse)
    # 磁场线 (波浪)
    for i, (dy, alpha) in enumerate([(0.3, 0.8), (0.0, 0.5), (-0.3, 0.3)]):
        t = np.linspace(0, 2, 30)
        ax.plot(cx + 0.7 + t * 0.6, cy + 1.3 + dy + 0.08 * np.sin(t * 6),
                color='#26A69A', lw=1.5, alpha=alpha, zorder=3)
    # 指示灯
    ax.add_patch(Circle((cx, cy+1.3), 0.08, fc='#4CAF50', ec='#2E7D32',
                         lw=0.8, zorder=4))


def draw_probe(ax, cx, cy):
    """画超声探头示意图"""
    # 手柄 (长矩形)
    ax.add_patch(FancyBboxPatch((cx-0.25, cy-0.5), 0.5, 1.6,
                                boxstyle="round,pad=0.06",
                                fc='#ECEFF1', ec='#607D8B', lw=1.2, zorder=3))
    # 探头头部 (梯形)
    head = Polygon([(cx-0.5, cy-0.5), (cx+0.5, cy-0.5),
                     (cx+0.35, cy-0.9), (cx-0.35, cy-0.9)],
                    closed=True, fc='#B0BEC5', ec='#455A64', lw=1.2, zorder=3)
    ax.add_patch(head)
    # 6DOF传感器 (小方块)
    ax.add_patch(FancyBboxPatch((cx-0.2, cy+0.6), 0.4, 0.35,
                                boxstyle="round,pad=0.02",
                                fc='#42A5F5', ec='#1565C0', lw=1.0, zorder=4))
    ax.text(cx, cy+0.78, '6DOF', fontsize=6, ha='center', va='center',
            color='white', fontweight='bold', zorder=5)
    # 线缆
    t = np.linspace(0, 1.2, 30)
    ax.plot(cx + 0.08 * np.sin(t * 8), cy + 1.1 + t * 0.5,
            color='#78909C', lw=1.5, zorder=3)


def draw_brain_icon(ax, cx, cy):
    """画大脑简化图标"""
    # 大脑轮廓 (两个半球)
    t = np.linspace(0, 2 * np.pi, 100)
    # 左半球
    ax.plot(cx - 0.1 + 0.55 * np.cos(t) * (np.cos(t) < 0.1).astype(float) +
            0.55 * np.cos(t) * (np.cos(t) >= 0.1).astype(float) * 0.8,
            cy + 0.5 * np.sin(t), color='#EF9A9A', lw=0)
    brain = matplotlib.patches.Ellipse((cx-0.15, cy), 0.8, 0.9,
                                        fc='#FFCDD2', ec='#E57373', lw=1.5, zorder=3)
    ax.add_patch(brain)
    brain2 = matplotlib.patches.Ellipse((cx+0.15, cy), 0.8, 0.9,
                                         fc='#FFCDD2', ec='#E57373', lw=1.5, zorder=3)
    ax.add_patch(brain2)
    # 中缝
    ax.plot([cx, cx], [cy-0.45, cy+0.45], color='#E57373', lw=1.0, zorder=4)
    # 脑回纹理
    for dy in [-0.15, 0.1, 0.3]:
        t = np.linspace(-0.3, 0.3, 20)
        ax.plot(cx + t, cy + dy + 0.05 * np.sin(t * 15),
                color='#E57373', lw=0.7, alpha=0.6, zorder=4)


# =================== 主图 ===================
def main():
    fig, ax = plt.subplots(figsize=(20, 7.5))
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ========================================
    # 第1列: 硬件设备 (磁场发射器 + 超声探头)
    # ========================================
    # 浅绿背景
    draw_rounded_box(ax, -0.5, -1.2, 4.0, 7.2, '#E8F5E9', '#A5D6A7', lw=1.0)
    ax.text(1.5, 5.7, '硬件设备', fontsize=14, ha='center',
            fontweight='bold', color='#2E7D32', zorder=6)

    # 磁场发射器
    draw_emitter(ax, 0.8, 3.0)
    ax.text(0.8, 2.0, '磁场发射器', fontsize=11, ha='center',
            fontweight='bold', color='#37474F', zorder=6)

    # 超声探头
    draw_probe(ax, 2.8, 3.5)
    ax.text(2.8, 2.0, '超声探头', fontsize=11, ha='center',
            fontweight='bold', color='#37474F', zorder=6)

    # 患者 (大脑图标)
    draw_brain_icon(ax, 1.8, 0.2)
    ax.text(1.8, -0.7, '患者脑组织', fontsize=10, ha='center',
            color='#C62828', fontweight='bold', zorder=6)

    # 探头 → 患者 虚线
    ax.annotate('', xy=(2.2, 0.7), xytext=(2.6, 2.5),
                arrowprops=dict(arrowstyle='->', color='#78909C',
                                lw=1.5, ls='--'))

    # ========================================
    # 第2列: 实时数据采集 (真实MRI/CT图像)
    # ========================================
    draw_rounded_box(ax, 4.2, -1.2, 3.8, 7.2, '#FFF8E1', '#FFE082', lw=1.0)
    ax.text(6.1, 5.7, '数据采集', fontsize=14, ha='center',
            fontweight='bold', color='#E65100', zorder=6)

    # 嵌入真实CT图像
    ct_path = os.path.join(DATA_DIR, 'CT', '01.png')
    if os.path.exists(ct_path):
        embed_image(ax, ct_path, 5.3, 3.5, zoom=0.35)
        ax.text(5.3, 2.0, 'CT图像', fontsize=11, ha='center',
                fontweight='bold', color='#37474F', zorder=6)

    # 嵌入真实MRI图像
    mri_path = os.path.join(DATA_DIR, 'MRI', '01.png')
    if os.path.exists(mri_path):
        embed_image(ax, mri_path, 7.0, 3.5, zoom=0.35)
        ax.text(7.0, 2.0, 'US图像', fontsize=11, ha='center',
                fontweight='bold', color='#37474F', zorder=6)

    # 6DOF数据标注
    ax.text(6.1, 0.8, '6DOF 位姿数据', fontsize=11, ha='center',
            fontweight='bold', color='#E65100', zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0',
                      ec='#FF9800', lw=1.0))
    # (坐标分量说明已省略)

    # 硬件 → 数据采集
    big_arrow(ax, 3.7, 3.0, 4.4, 3.0, '#546E7A', 3.0)

    # ========================================
    # 第3列: 坐标变换与优化
    # ========================================
    draw_rounded_box(ax, 8.6, -1.2, 3.8, 7.2, '#E3F2FD', '#90CAF9', lw=1.0)
    ax.text(10.5, 5.7, '坐标变换与优化', fontsize=14, ha='center',
            fontweight='bold', color='#1565C0', zorder=6)

    # Rodrigues旋转
    draw_rounded_box(ax, 9.0, 3.5, 3.0, 1.5, '#BBDEFB', '#1E88E5', lw=1.2)
    ax.text(10.5, 4.55, 'Rodrigues旋转', fontsize=12, ha='center',
            fontweight='bold', color='#0D47A1', zorder=6)
    ax.text(10.5, 3.95,
            r'$\mathbf{P}_{g} = \mathbf{R} \cdot \mathbf{P}_{l} + \mathbf{t}$',
            fontsize=11, ha='center', color='#1B5E20', zorder=6)

    # 双线性插值
    draw_rounded_box(ax, 9.0, 1.5, 3.0, 1.5, '#BBDEFB', '#1E88E5', lw=1.2)
    ax.text(10.5, 2.55, '双线性插值', fontsize=12, ha='center',
            fontweight='bold', color='#0D47A1', zorder=6)
    ax.text(10.5, 1.95, '像素→3D坐标', fontsize=11, ha='center',
            fontweight='bold', color='#37474F', zorder=6)

    # 数据优化
    draw_rounded_box(ax, 9.0, -0.5, 3.0, 1.5, '#BBDEFB', '#1E88E5', lw=1.2)
    ax.text(10.5, 0.55, '数据优化', fontsize=12, ha='center',
            fontweight='bold', color='#0D47A1', zorder=6)
    ax.text(10.5, -0.05, '最小二乘+卡尔曼滤波', fontsize=10, ha='center',
            fontweight='bold', color='#37474F', zorder=6)

    # 内部箭头
    ax.annotate('', xy=(10.5, 3.1), xytext=(10.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))
    ax.annotate('', xy=(10.5, 1.1), xytext=(10.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    # 数据采集 → 坐标变换
    big_arrow(ax, 8.2, 3.0, 8.8, 3.0, '#546E7A', 3.0)

    # ========================================
    # 第4列: 输出 (定位后图像 + 后续应用)
    # ========================================
    draw_rounded_box(ax, 13.0, -1.2, 7.5, 7.2, '#F3E5F5', '#CE93D8', lw=1.0)
    ax.text(16.75, 5.7, '输出与应用', fontsize=14, ha='center',
            fontweight='bold', color='#6A1B9A', zorder=6)

    # 定位后US图像 (嵌入真实配准结果)
    draw_rounded_box(ax, 13.5, 2.3, 3.2, 3.0, '#E1BEE7', '#8E24AA', lw=1.5)
    ax.text(15.1, 5.0, '定位后US图像', fontsize=12, ha='center',
            fontweight='bold', color='#4A148C', zorder=6)

    # 嵌入真实配准结果图
    deformed_path = os.path.join(DATA_DIR, 'MRI_deformed', '01_deformed_1.png')
    if os.path.exists(deformed_path):
        embed_image(ax, deformed_path, 15.1, 3.5, zoom=0.38)

    ax.text(15.1, 2.5, '含3D空间坐标', fontsize=11, ha='center',
            fontweight='bold', color='#6A1B9A', zorder=6)

    # 后续应用1: MRI-US配准
    draw_rounded_box(ax, 17.3, 3.2, 2.8, 1.5, '#F3E5F5', '#8E24AA', lw=1.2, zorder=3)
    ax.text(18.7, 4.3, 'MRI-US', fontsize=11, ha='center',
            fontweight='bold', color='#4A148C', zorder=6)
    ax.text(18.7, 3.7, '配准', fontsize=11, ha='center',
            fontweight='bold', color='#4A148C', zorder=6)

    # 后续应用2: 误差先验建模
    draw_rounded_box(ax, 17.3, 0.8, 2.8, 1.5, '#FFEBEE', '#E53935', lw=1.2, zorder=3)
    ax.text(18.7, 1.9, '误差先验', fontsize=11, ha='center',
            fontweight='bold', color='#B71C1C', zorder=6)
    ax.text(18.7, 1.3, '建模', fontsize=11, ha='center',
            fontweight='bold', color='#B71C1C', zorder=6)

    # 定位后US → 后续
    ax.annotate('', xy=(17.3, 4.0), xytext=(16.7, 4.0),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=2.0))
    ax.annotate('', xy=(17.3, 1.5), xytext=(16.7, 2.8),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.0,
                                connectionstyle='arc3,rad=0.2'))

    # 坐标变换 → 输出
    big_arrow(ax, 12.2, 3.0, 13.2, 3.0, '#546E7A', 3.0)

    # (精度标注已省略)

    # ===== 保存 =====
    save_path = os.path.join('figures', 'fig_magnetic_positioning_workflow.png')
    os.makedirs('figures', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✓ 保存至: {save_path}")


if __name__ == '__main__':
    main()
