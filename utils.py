import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import cv2


def enhanced_flow_visualization(flow_field, output_types=None, background_img=None):
    """
    创建变形场的增强可视化

    参数:
        flow_field: 变形场 (2, H, W) 或 (B, 2, H, W)，表示 x 和 y 方向的位移
        output_types: 要生成的可视化类型列表，可选值包括：
                    'hsv' - HSV颜色编码的方向和幅度
                    'magnitude' - 幅度热图
                    'quiver' - 向量场可视化
                    'jacobian' - 雅可比行列式可视化
                    'contour' - 等高线可视化
                    'divergence' - 散度可视化
                    'curl' - 旋度可视化
                    如果为None，则生成所有类型
        background_img: 可选的背景图像 (H, W) 用于叠加可视化

    返回:
        dict: 包含不同类型可视化的字典，键为类型名称，值为RGB图像数组 (H, W, 3)
    """
    # 确保输入是正确的形状
    if torch.is_tensor(flow_field):
        if flow_field.dim() == 4:  # (B, 2, H, W)
            flow_field = flow_field[0]  # 取第一个批次
        flow_field = flow_field.detach().cpu().numpy()
    elif isinstance(flow_field, np.ndarray) and flow_field.ndim == 4:
        flow_field = flow_field[0]  # 取第一个批次

    # 确保是 (2, H, W) 形状
    if flow_field.shape[0] != 2:
        if flow_field.shape[-1] == 2:  # (H, W, 2)
            flow_field = flow_field.transpose(2, 0, 1)
        else:
            raise ValueError("无效的流场形状，应为(2, H, W)或(B, 2, H, W)")

    # 提取x和y方向的流场
    u = flow_field[0]  # x方向流场
    v = flow_field[1]  # y方向流场

    H, W = u.shape

    # 定义输出类型
    if output_types is None:
        output_types = ['hsv', 'magnitude', 'quiver', 'jacobian', 'contour', 'divergence', 'curl']

    # 准备结果字典
    results = {}

    # 计算常用的变形场特征
    # 1. 幅度 - 表示位移的大小
    magnitude = np.sqrt(u ** 2 + v ** 2)
    max_magnitude = np.max(magnitude) if np.max(magnitude) > 0 else 1.0

    # 2. 方向 - 表示位移的方向
    angle = np.arctan2(v, u)

    # 创建背景图（如果提供）
    if background_img is not None:
        if torch.is_tensor(background_img):
            background_img = background_img.detach().cpu().numpy()
            if background_img.ndim == 3 and background_img.shape[0] == 1:
                background_img = background_img[0]  # 取单通道

        # 归一化背景到 [0, 1]
        background_img = (background_img - np.min(background_img)) / (
                    np.max(background_img) - np.min(background_img) + 1e-8)
        background_rgb = np.stack([background_img] * 3, axis=-1) * 0.5  # 减小背景亮度以便于观察叠加效果
    else:
        background_rgb = np.zeros((H, W, 3))

    # 根据需要生成不同类型的可视化
    # 1. HSV颜色编码的流场可视化 - 颜色表示方向，亮度表示幅度
    if 'hsv' in output_types:
        # 创建HSV图像
        hsv = np.zeros((H, W, 3), dtype=np.float32)
        hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # 色调 - 方向 [0, 1]
        hsv[..., 1] = np.ones_like(angle)  # 饱和度 - 全饱和
        hsv[..., 2] = magnitude / max_magnitude  # 明度 - 幅度

        # 转换为RGB
        rgb_hsv = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 添加等高线以突出轮廓
        norm_magnitude = magnitude / max_magnitude

        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_hsv)

        # 为这个特定图像创建网格，确保大小匹配
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        # 添加幅度等高线，使用白色细线
        plt.contour(X, Y, norm_magnitude, levels=10, colors='white', linewidths=0.5, alpha=0.7)
        plt.axis('off')

        # 保存到内存而不是显示
        fig = plt.gcf()
        fig.canvas.draw()
        plt_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()

        # 调整大小以匹配原始尺寸
        hsv_result = cv2.resize(plt_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        results['hsv'] = hsv_result

    # 2. 幅度热图可视化 - 使用热图显示位移幅度
    if 'magnitude' in output_types:
        # 创建自定义的彩虹颜色映射
        colors_list = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        rainbow_cmap = LinearSegmentedColormap.from_list("rainbow", colors_list)

        plt.figure(figsize=(10, 8))
        plt.imshow(background_rgb)
        # 添加半透明的幅度热图
        plt.imshow(magnitude, cmap=rainbow_cmap, alpha=0.7)

        # 重新创建网格以确保大小匹配
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        # 添加等高线
        plt.contour(X, Y, magnitude, levels=15, colors='white', linewidths=0.5, alpha=0.9)
        plt.colorbar(label='Displacement Magnitude')
        plt.title('Deformation Field Magnitude')
        plt.axis('off')

        # 保存到内存而不是显示
        fig = plt.gcf()
        fig.canvas.draw()
        plt_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()

        # 调整大小以匹配原始尺寸
        magnitude_result = cv2.resize(plt_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        results['magnitude'] = magnitude_result

    # 3. 向量场可视化 - 使用箭头显示变形方向和幅度
    if 'quiver' in output_types:
        # 对网格进行下采样以避免箭头过密
        step = max(1, min(H, W) // 40)  # 根据图像大小自适应步长

        plt.figure(figsize=(10, 8))
        plt.imshow(background_rgb)

        # 创建下采样的网格 - 注意这里是有意下采样的
        Y_down, X_down = np.mgrid[:H:step, :W:step]
        U = u[::step, ::step]
        V = v[::step, ::step]

        # 为向量场的幅度创建颜色映射
        M = np.sqrt(U ** 2 + V ** 2)

        # 使用quiver绘制箭头，颜色编码表示幅度 - 这里网格和数据已经匹配了
        plt.quiver(X_down, Y_down, U, V, M, cmap='jet', width=0.001, scale=50, alpha=0.8)
        plt.colorbar(label='Vector Magnitude')
        plt.title('Deformation Field Vectors')
        plt.axis('off')

        # 保存到内存而不是显示
        fig = plt.gcf()
        fig.canvas.draw()
        plt_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()

        # 调整大小以匹配原始尺寸
        quiver_result = cv2.resize(plt_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        results['quiver'] = quiver_result

    # 4. 雅可比行列式可视化 - 显示局部体积变化
    if 'jacobian' in output_types:
        # 计算雅可比矩阵的行列式
        # 使用有限差分计算梯度
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)

        # x方向梯度
        du_dx[:, :-1] = u[:, 1:] - u[:, :-1]
        dv_dx[:, :-1] = v[:, 1:] - v[:, :-1]

        # y方向梯度
        du_dy[:-1, :] = u[1:, :] - u[:-1, :]
        dv_dy[:-1, :] = v[1:, :] - v[:-1, :]

        # 计算雅可比行列式: (1+du/dx)*(1+dv/dy) - (du/dy)*(dv/dx)
        jacobian_det = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx

        # 使用发散的颜色映射来显示膨胀(>1)和收缩(<1)
        # 设置范围从0.5到1.5，使1.0(无变化)为白色
        divnorm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)

        plt.figure(figsize=(10, 8))
        plt.imshow(background_rgb)
        # 添加半透明的雅可比行列式可视化
        plt.imshow(jacobian_det, cmap='coolwarm', norm=divnorm, alpha=0.8)
        plt.colorbar(label='Jacobian Determinant')

        # 为雅可比行列式创建正确大小的网格
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        plt.contour(X, Y, jacobian_det, levels=[0.7, 0.9, 1.0, 1.1, 1.3],
                    colors=['blue', 'lightblue', 'white', 'pink', 'red'],
                    linewidths=0.5, alpha=0.9)
        plt.title('Jacobian Determinant (Volume Change)')
        plt.axis('off')

        # 保存到内存而不是显示
        fig = plt.gcf()
        fig.canvas.draw()
        plt_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()

        # 调整大小以匹配原始尺寸
        jacobian_result = cv2.resize(plt_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        results['jacobian'] = jacobian_result

    # 5. 等高线可视化 - 使用彩色等高线显示变形场
    if 'contour' in output_types:
        plt.figure(figsize=(10, 8))
        plt.imshow(background_rgb)

        # 为等高线创建正确大小的网格
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        # 创建u和v方向的等高线
        u_levels = np.linspace(np.min(u), np.max(u), 15)
        v_levels = np.linspace(np.min(v), np.max(v), 15)

        # 绘制u方向等高线
        u_contour = plt.contour(X, Y, u, levels=u_levels, cmap='cool', linewidths=1, alpha=0.7)
        plt.colorbar(u_contour, label='X-displacement')

        # 绘制v方向等高线
        v_contour = plt.contour(X, Y, v, levels=v_levels, cmap='autumn', linewidths=1, alpha=0.7)
        plt.colorbar(v_contour, label='Y-displacement')

        plt.title('Deformation Field Contours')
        plt.axis('off')

        # 保存到内存而不是显示
        fig = plt.gcf()
        fig.canvas.draw()
        plt_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()

        # 调整大小以匹配原始尺寸
        contour_result = cv2.resize(plt_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        results['contour'] = contour_result

    # 6. 散度可视化 - 显示局部膨胀/收缩
    if 'divergence' in output_types:
        # 计算散度: ∇·F = ∂u/∂x + ∂v/∂y
        divergence = du_dx + dv_dy

        # 使用发散的颜色映射来显示膨胀(正)和收缩(负)
        plt.figure(figsize=(10, 8))
        plt.imshow(background_rgb)
        # 添加半透明的散度可视化
        divergence_map = plt.imshow(divergence, cmap='RdBu_r', alpha=0.8)
        plt.colorbar(divergence_map, label='Divergence')

        # 为散度创建正确大小的网格
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        # 添加散度等高线
        plt.contour(X, Y, divergence, levels=15, colors='black', linewidths=0.5, alpha=0.5)
        plt.title('Divergence (Expansion/Contraction)')
        plt.axis('off')

        # 保存到内存而不是显示
        fig = plt.gcf()
        fig.canvas.draw()
        plt_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()

        # 调整大小以匹配原始尺寸
        divergence_result = cv2.resize(plt_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        results['divergence'] = divergence_result

    # 7. 旋度可视化 - 显示局部旋转
    if 'curl' in output_types:
        # 计算旋度: ∇×F = ∂v/∂x - ∂u/∂y (对于2D流场，旋度是标量)
        curl = dv_dx - du_dy

        plt.figure(figsize=(10, 8))
        plt.imshow(background_rgb)
        # 添加半透明的旋度可视化
        curl_map = plt.imshow(curl, cmap='PiYG', alpha=0.8)
        plt.colorbar(curl_map, label='Curl')

        # 为旋度创建正确大小的网格
        X, Y = np.meshgrid(np.arange(W), np.arange(H))

        # 添加旋度等高线
        plt.contour(X, Y, curl, levels=15, colors='black', linewidths=0.5, alpha=0.5)
        plt.title('Curl (Rotation)')
        plt.axis('off')

        # 保存到内存而不是显示
        fig = plt.gcf()
        fig.canvas.draw()
        plt_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()

        # 调整大小以匹配原始尺寸
        curl_result = cv2.resize(plt_img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        results['curl'] = curl_result

    return results


def create_rainbow_flow_visualization(flow_field, background_img=None, alpha=0.7):
    """
    创建更加丰富多彩的变形场可视化，突出显示轮廓

    参数:
        flow_field: 变形场 (2, H, W) 或 (B, 2, H, W)
        background_img: 可选的背景图像
        alpha: 可视化的透明度

    返回:
        rgb_image: RGB图像 (H, W, 3)
    """
    # 确保输入是正确的形状
    if torch.is_tensor(flow_field):
        if flow_field.dim() == 4:  # (B, 2, H, W)
            flow_field = flow_field[0]  # 取第一个批次
        flow_field = flow_field.detach().cpu().numpy()
    elif isinstance(flow_field, np.ndarray) and flow_field.ndim == 4:
        flow_field = flow_field[0]  # 取第一个批次

    # 确保是 (2, H, W) 形状
    if flow_field.shape[0] != 2:
        if flow_field.shape[-1] == 2:  # (H, W, 2)
            flow_field = flow_field.transpose(2, 0, 1)
        else:
            raise ValueError("无效的流场形状，应为(2, H, W)或(B, 2, H, W)")

    # 提取x和y方向的流场
    u = flow_field[0]  # x方向流场
    v = flow_field[1]  # y方向流场

    H, W = u.shape

    # 计算流场特征
    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)

    # 归一化幅度和角度
    max_magnitude = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    norm_magnitude = magnitude / max_magnitude
    norm_angle = (angle + np.pi) / (2 * np.pi)  # 映射到[0,1]

    # 创建背景
    if background_img is not None:
        if torch.is_tensor(background_img):
            background_img = background_img.detach().cpu().numpy()
            if background_img.ndim == 3 and background_img.shape[0] == 1:
                background_img = background_img[0]  # 取单通道

        # 归一化背景到 [0, 1]
        background_img = (background_img - np.min(background_img)) / (
                    np.max(background_img) - np.min(background_img) + 1e-8)
        background = np.stack([background_img] * 3, axis=-1) * 0.3  # 减小背景亮度
    else:
        background = np.zeros((H, W, 3))

    # 创建彩虹色图像 - 使用角度和幅度共同决定颜色
    hsv = np.zeros((H, W, 3), dtype=np.float32)
    hsv[..., 0] = norm_angle  # 色调 - 角度
    hsv[..., 1] = 0.9 + 0.1 * norm_magnitude  # 饱和度 - 稍微受幅度影响
    hsv[..., 2] = 0.2 + 0.8 * norm_magnitude  # 明度 - 主要由幅度决定

    # 转换为RGB
    flow_rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 添加轮廓线增强效果
    plt.figure(figsize=(10, 8))
    plt.imshow(background)
    plt.imshow(flow_rgb, alpha=alpha)

    # 创建对应大小的网格
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # 添加u和v方向的等高线，使用不同颜色
    u_contour = plt.contour(X, Y, u, levels=10, colors='white', linewidths=0.5, alpha=0.7)
    v_contour = plt.contour(X, Y, v, levels=10, colors='yellow', linewidths=0.5, alpha=0.7)

    # 添加幅度等高线
    mag_contour = plt.contour(X, Y, magnitude, levels=8, colors='cyan', linewidths=0.7, alpha=0.8)

    # 添加边缘增强
    edges = cv2.Canny((norm_magnitude * 255).astype(np.uint8), 50, 150)
    plt.contour(X, Y, edges, levels=[1], colors='white', linewidths=1, alpha=0.9)

    plt.axis('off')

    # 保存到内存
    fig = plt.gcf()
    fig.canvas.draw()
    result_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close()

    # 调整大小以匹配原始尺寸
    result_img = cv2.resize(result_img, (W, H), interpolation=cv2.INTER_LANCZOS4)

    return result_img


def visualize_deformation_fields(flow_field, fixed_img=None, moving_img=None, registered_img=None, mode='rainbow'):
    """
    可视化变形场及相关图像，适合集成到PyTorch训练/测试循环中

    参数:
        flow_field: 变形场 (B, 2, H, W) 或 (2, H, W)
        fixed_img: 固定图像 (可选)
        moving_img: 移动图像 (可选)
        registered_img: 配准后的图像 (可选)
        mode: 可视化模式，可选 'rainbow', 'hsv', 'contour', 'all'

    返回:
        visualization_dict: 包含不同可视化结果的字典
    """
    if torch.is_tensor(flow_field):
        if flow_field.dim() == 4:  # (B, 2, H, W)
            flow_field = flow_field[0]  # 取第一个批次

    results = {}

    # 根据选择的模式生成不同的可视化
    if mode == 'rainbow' or mode == 'all':
        rainbow_viz = create_rainbow_flow_visualization(flow_field, background_img=fixed_img)
        results['rainbow'] = torch.from_numpy(rainbow_viz.transpose(2, 0, 1) / 255.0).float()

    if mode == 'hsv' or mode == 'all':
        viz_dict = enhanced_flow_visualization(flow_field, output_types=['hsv'], background_img=fixed_img)
        results['hsv'] = torch.from_numpy(viz_dict['hsv'].transpose(2, 0, 1) / 255.0).float()

    if mode == 'contour' or mode == 'all':
        viz_dict = enhanced_flow_visualization(flow_field, output_types=['contour'], background_img=fixed_img)
        results['contour'] = torch.from_numpy(viz_dict['contour'].transpose(2, 0, 1) / 255.0).float()

    if mode == 'all':
        all_types = ['magnitude', 'quiver', 'jacobian', 'divergence', 'curl']
        viz_dict = enhanced_flow_visualization(flow_field, output_types=all_types, background_img=fixed_img)
        for key, val in viz_dict.items():
            if key != 'combined':  # 忽略组合图像以避免重复
                results[key] = torch.from_numpy(val.transpose(2, 0, 1) / 255.0).float()

    # 如果提供了图像数据，添加配准前后对比可视化
    if fixed_img is not None and moving_img is not None and registered_img is not None:
        # 创建对比图像
        plt.figure(figsize=(15, 5))

        # 将所有图像转换为numpy数组
        fixed_np = fixed_img[0].cpu().numpy() if torch.is_tensor(fixed_img) else fixed_img
        moving_np = moving_img[0].cpu().numpy() if torch.is_tensor(moving_img) else moving_img
        registered_np = registered_img[0].cpu().numpy() if torch.is_tensor(registered_img) else registered_img

        # 归一化到[0,1]
        for img in [fixed_np, moving_np, registered_np]:
            if np.max(img) > np.min(img):
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # 添加配准前后对比和差异图
        plt.subplot(1, 3, 1)
        plt.imshow(fixed_np, cmap='gray')
        plt.title('Fixed Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(moving_np, cmap='gray')
        plt.title('Moving Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(registered_np, cmap='gray')
        plt.title('Registered Image')
        plt.axis('off')

        plt.tight_layout()

        # 保存到内存
        fig = plt.gcf()
        fig.canvas.draw()
        comparison_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()

        results['comparison'] = torch.from_numpy(comparison_img.transpose(2, 0, 1) / 255.0).float()

    return results


# 用于转换回原始visualize_flow函数接口的兼容函数
def visualize_flow_enhanced(flow_field, convert_to_rgb=True, mode='rainbow'):
    """
    增强版的可视化函数，兼容原有接口

    参数:
        flow_field: 变形场 (2, H, W)
        convert_to_rgb: 是否转换为RGB图像
        mode: 可视化模式

    返回:
        rgb_tensor: RGB图像张量 (3, H, W)
    """
    if mode == 'rainbow':
        # 使用彩虹型可视化
        rgb_img = create_rainbow_flow_visualization(flow_field)
        return torch.from_numpy(rgb_img.transpose(2, 0, 1) / 255.0).float()
    elif mode == 'hsv':
        # 使用HSV型可视化
        viz_dict = enhanced_flow_visualization(flow_field, output_types=['hsv'])
        return torch.from_numpy(viz_dict['hsv'].transpose(2, 0, 1) / 255.0).float()
    else:
        # 使用多种可视化并组合
        viz_dict = enhanced_flow_visualization(flow_field, output_types=[mode])
        return torch.from_numpy(viz_dict[mode].transpose(2, 0, 1) / 255.0).float()
def save_image(imgs, titles, save_path, cmap='gray', use_3ch_gray=False):
    """
    保存图像网格

    参数:
        imgs: 图像张量列表 [(C, H, W), ...]
        titles: 对应的标题列表
        save_path: 保存路径
        cmap: 颜色映射，默认为灰度
        use_3ch_gray: 是否将单通道灰度图像转换为三通道灰度图像
    """
    n = len(imgs)
    plt.figure(figsize=(4 * n, 4))

    processed_imgs = []

    # 预处理所有图像
    for img in imgs:
        if torch.is_tensor(img):
            # 处理单通道图像
            if img.dim() == 3 and img.size(0) == 1:
                img_np = img[0].detach().cpu().numpy()

                # 归一化到[0,1]范围
                if img_np.max() > img_np.min():
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

                # 如果需要三通道灰度图像
                if use_3ch_gray:
                    # 使用灰度颜色映射将单通道图像转换为RGB
                    cm_gray = plt.cm.get_cmap('gray')
                    img_np = cm_gray(img_np)[:, :, :3]  # 去掉alpha通道
                    processed_imgs.append(img_np)
                else:
                    processed_imgs.append(img_np)

            # 处理三通道图像
            elif img.dim() == 3 and img.size(0) == 3:
                img_np = img.permute(1, 2, 0).detach().cpu().numpy()

                # 归一化到[0,1]范围
                if img_np.max() > 1.0 or img_np.min() < 0.0:
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

                processed_imgs.append(img_np)

            # 处理其他情况
            else:
                processed_imgs.append(img.detach().cpu().numpy())
        else:
            # 如果已经是numpy数组
            processed_imgs.append(img)

    # 显示和保存预处理后的图像
    for i, (img, title) in enumerate(zip(processed_imgs, titles)):
        plt.subplot(1, n, i + 1)

        # 显示图像
        if img.ndim == 2 or (img.ndim == 3 and not use_3ch_gray):
            plt.imshow(img, cmap=cmap if img.ndim == 2 else None)
        else:
            plt.imshow(img)

        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存图像到 {save_path}")


def visualize_flow(flow_field, convert_to_rgb=True):
    """
    可视化变形场

    参数:
        flow_field: 变形场 (2, H, W)
        convert_to_rgb: 是否转换为RGB图像

    返回:
        可视化的图像张量
    """
    # 计算变形场的幅度和方向
    u = flow_field[0].numpy()
    v = flow_field[1].numpy()

    # 计算幅度和角度
    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)

    # 归一化幅度到[0, 1]范围
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    # 创建HSV图像（色调表示方向，饱和度为1，明度表示幅度）
    h = (angle + np.pi) / (2 * np.pi)  # 将角度映射到[0, 1]
    s = np.ones_like(h)
    v = magnitude

    # 创建HSV图像
    hsv = np.stack([h, s, v], axis=2)

    if convert_to_rgb:
        # 转换为RGB
        rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        # 转换为PyTorch张量
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        return rgb_tensor
    else:
        # 如果不需要转换为RGB，则返回HSV图像
        return torch.from_numpy(hsv.transpose(2, 0, 1)).float()


def save_checkpoint(state, is_best, save_path):
    """
    保存检查点

    参数:
        state: 包含模型状态等的字典
        is_best: 是否是最佳模型
        save_path: 保存路径
    """
    checkpoint_path = os.path.join(save_path, 'checkpoint.pth')
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(save_path, 'best_model.pth')
        torch.save(state, best_path)
        print(f"已保存最佳模型到 {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    加载检查点

    参数:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器 (可选)

    返回:
        开始的epoch和最佳损失
    """
    if os.path.isfile(checkpoint_path):
        print(f"从 {checkpoint_path} 加载检查点")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        best_loss = checkpoint.get('best_loss', float('inf'))

        return start_epoch, best_loss
    else:
        print(f"未找到检查点: {checkpoint_path}")
        return 0, float('inf')


def compute_dice_score(pred, target, threshold=0.5):
    """
    计算Dice系数

    参数:
        pred: 预测图像 (B, 1, H, W)
        target: 目标图像 (B, 1, H, W)
        threshold: 二值化阈值

    返回:
        Dice系数值
    """
    # 将图像转换为二值图像
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # 计算交集
    intersection = (pred_binary * target_binary).sum()

    # 计算两个集合的大小
    size_pred = pred_binary.sum()
    size_target = target_binary.sum()

    # 计算Dice系数
    dice = (2.0 * intersection) / (size_pred + size_target + 1e-6)

    return dice.item()


def apply_colormap(gray_img, cmap_name='jet'):
    """
    将灰度图像转换为彩色图像

    参数:
        gray_img: 灰度图像张量 (1, H, W)
        cmap_name: matplotlib颜色映射名称

    返回:
        彩色图像张量 (3, H, W)
    """
    if torch.is_tensor(gray_img):
        # 确保是单通道图像
        if gray_img.ndim == 3 and gray_img.shape[0] == 1:
            gray_np = gray_img[0].detach().cpu().numpy()
        else:
            gray_np = gray_img.detach().cpu().numpy()
    else:
        gray_np = gray_img

    # 归一化到[0, 1]
    if gray_np.max() > gray_np.min():
        gray_np = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min())

    # 应用颜色映射
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(gray_np)

    # 去掉alpha通道并转换为(C, H, W)格式
    colored = colored[..., :3].transpose(2, 0, 1)

    return torch.from_numpy(colored).float()


def gray_to_3channel(gray_img):
    """
    将单通道灰度图像转换为三通道灰度图像

    参数:
        gray_img: 灰度图像张量 (1, H, W)

    返回:
        三通道灰度图像张量 (3, H, W)
    """
    if torch.is_tensor(gray_img):
        # 确保是单通道图像
        if gray_img.ndim == 3 and gray_img.shape[0] == 1:
            gray_np = gray_img[0].detach().cpu().numpy()
        else:
            gray_np = gray_img.detach().cpu().numpy()
    else:
        gray_np = gray_img

    # 归一化到[0, 1]，如果需要的话
    if gray_np.max() > 1.0 or gray_np.min() < 0.0:
        gray_np = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min())

    # 创建三通道灰度图像
    gray_3ch = np.stack([gray_np, gray_np, gray_np], axis=0)

    return torch.from_numpy(gray_3ch).float()