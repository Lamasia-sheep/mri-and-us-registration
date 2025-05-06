import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2


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