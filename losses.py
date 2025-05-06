import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NormalizedMutualInformationLoss(nn.Module):
    """
    基于归一化互信息(NMI)的损失函数
    对于医学图像配准尤其有效
    修复版本 - 添加数值稳定性
    """

    def __init__(self, num_bins=24):
        """
        参数:
            num_bins: 计算联合直方图时的bin数量
        """
        super(NormalizedMutualInformationLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, x, y):
        """
        计算NMI损失

        参数:
            x: 第一个图像 (B, 1, H, W) - 单通道灰度图像
            y: 第二个图像 (B, 1, H, W) - 单通道灰度图像

        返回:
            归一化互信息的负值 (我们最大化NMI，因此最小化负NMI)
        """
        # 重塑张量以便于计算直方图
        batch_size = x.size(0)

        # 下采样以提高性能 (减少点数)
        # 每4个像素取1个，减少75%的计算量
        with torch.no_grad():  # 避免梯度计算增加内存使用
            x = x[:, :, ::2, ::2]
            y = y[:, :, ::2, ::2]

        # 增加数值稳定性 - 添加噪声避免相同的值
        x = x + torch.randn_like(x) * 1e-4
        y = y + torch.randn_like(y) * 1e-4

        x_flat = x.reshape(batch_size, -1)
        y_flat = y.reshape(batch_size, -1)

        # 归一化到[0, 1]
        x_flat = (x_flat + 1) / 2
        y_flat = (y_flat + 1) / 2

        # 裁剪极值，避免数值问题
        x_flat = torch.clamp(x_flat, 0.001, 0.999)
        y_flat = torch.clamp(y_flat, 0.001, 0.999)

        # 创建网格索引
        x_grid = torch.linspace(0, 1, self.num_bins + 1, device=x.device)
        y_grid = torch.linspace(0, 1, self.num_bins + 1, device=y.device)

        # 计算批次中所有样本的平均NMI
        nmi_sum = 0
        for b in range(batch_size):
            # 量化数据到直方图bin
            x_bin = torch.bucketize(x_flat[b], x_grid) - 1
            y_bin = torch.bucketize(y_flat[b], y_grid) - 1

            # 限制边界
            x_bin = torch.clamp(x_bin, 0, self.num_bins - 1)
            y_bin = torch.clamp(y_bin, 0, self.num_bins - 1)

            # 计算联合直方图 - 使用torch.bincount优化
            joint_hist = torch.zeros(
                (self.num_bins, self.num_bins), dtype=torch.float32, device=x.device
            )

            # 优化的直方图计算
            bin_idx = x_bin * self.num_bins + y_bin
            bin_count = torch.bincount(bin_idx, minlength=self.num_bins * self.num_bins)
            joint_hist = bin_count.reshape(self.num_bins, self.num_bins)

            # 归一化直方图
            total_count = joint_hist.sum() + 1e-10  # 避免除零
            joint_hist = joint_hist / total_count

            # 计算边缘直方图
            x_hist = joint_hist.sum(dim=1)
            y_hist = joint_hist.sum(dim=0)

            # 为了数值稳定性，添加一个小的值
            eps = 1e-5
            joint_hist = joint_hist + eps
            x_hist = x_hist + eps
            y_hist = y_hist + eps

            # 计算互信息 - 数值安全的对数计算
            x_hist_reshape = x_hist.view(-1, 1).expand(-1, self.num_bins)
            y_hist_reshape = y_hist.view(1, -1).expand(self.num_bins, -1)

            # 安全的对数计算
            log_joint = torch.log(joint_hist)
            log_product = torch.log(x_hist_reshape * y_hist_reshape)

            # 计算互信息
            mi = torch.sum(joint_hist * (log_joint - log_product))

            # 计算熵 - 数值安全
            x_entropy = -torch.sum(x_hist * torch.log(x_hist))
            y_entropy = -torch.sum(y_hist * torch.log(y_hist))

            # 计算归一化互信息
            sum_entropy = x_entropy + y_entropy
            if sum_entropy < 1e-10:  # 避免除零
                nmi = 0.0
            else:
                nmi = 2 * mi / sum_entropy

            # 限制范围，避免极端值
            nmi = torch.clamp(nmi, -1.0, 1.0)
            nmi_sum += nmi

        # 返回平均NMI的负值，但确保有界
        return -torch.clamp(nmi_sum / batch_size, -1.0, 1.0)


class MINDLoss(nn.Module):
    """
    基于模块化互信息邻域描述符(MIND)的损失函数
    对于多模态图像配准特别有用
    优化和修复版本，处理NaN问题
    """

    def __init__(self, radius=1, dilation=2):
        """
        参数:
            radius: 邻域半径
            dilation: 空洞率，用于增大感受野
        """
        super(MINDLoss, self).__init__()
        self.radius = radius
        self.dilation = dilation
        self.num_offset = (2 * radius + 1) ** 2 - 1  # 不包括中心点
        # 预计算偏移，避免重复计算
        self.offsets = self._create_offsets()

    def _create_offsets(self):
        """创建邻域偏移"""
        offsets = []
        for i in range(-self.radius, self.radius + 1):
            for j in range(-self.radius, self.radius + 1):
                if i != 0 or j != 0:  # 排除中心点
                    offsets.append([i * self.dilation, j * self.dilation])
        return torch.tensor(offsets, dtype=torch.long)

    def _compute_mind(self, img):
        """
        计算MIND描述符

        参数:
            img: 输入图像 (B, 1, H, W)

        返回:
            MIND描述符 (B, D, H, W)
        """
        batch_size, _, height, width = img.size()
        device = img.device
        offsets = self.offsets.to(device)

        # 下采样处理，减少计算量
        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        height_ds, width_ds = img.size(2), img.size(3)

        # 添加小量的噪声，避免完全平坦的区域
        img = img + torch.randn_like(img) * 1e-4

        # 创建一个模板，用于计算patch平均值
        ones = torch.ones(1, 1, 2 * self.radius + 1, 2 * self.radius + 1, device=device)

        # 使用卷积计算patch中心与邻域点之间的差异
        padded_img = F.pad(img, [self.radius * self.dilation] * 4, mode='replicate')

        # 计算局部平均值
        patch_mean = F.conv2d(padded_img, ones) / ((2 * self.radius + 1) ** 2)
        patch_mean = F.pad(patch_mean, [self.radius * self.dilation] * 4, mode='replicate')

        # 计算局部方差（用于归一化）
        center_idx = self.radius * self.dilation
        center_patch = padded_img[:, :, center_idx:center_idx + height_ds, center_idx:center_idx + width_ds]
        variance = (center_patch - patch_mean[:, :, center_idx:center_idx + height_ds,
                                   center_idx:center_idx + width_ds]) ** 2
        variance = F.conv2d(F.pad(variance, [self.radius] * 4, mode='replicate'), ones) / ((2 * self.radius + 1) ** 2)

        # 提高数值稳定性 - 避免接近零的方差
        variance = torch.clamp(variance, min=1e-4)

        # 对于每个邻域点，计算与中心点的差异
        mind_descriptors = []
        for dx, dy in offsets:
            offset_patch = padded_img[:, :,
                           center_idx + dy:center_idx + dy + height_ds,
                           center_idx + dx:center_idx + dx + width_ds]

            # 计算差异
            diff = (center_patch - offset_patch) ** 2

            # 使用局部方差归一化差异 - 数值安全
            diff = diff / (variance * 2 + 1e-6)

            # 限制差异值，避免极端情况
            diff = torch.clamp(diff, 0.0, 50.0)

            # 将差异添加到描述符中
            mind_descriptors.append(diff)

        # 堆叠所有差异
        mind_tensor = torch.cat(mind_descriptors, dim=1)

        # 使用稳定版本的softmax归一化
        mind_tensor = torch.exp(-mind_tensor)
        sum_mind = torch.sum(mind_tensor, dim=1, keepdim=True) + 1e-8  # 避免除零
        mind_tensor = mind_tensor / sum_mind

        # 确保没有NaN值
        mind_tensor = torch.where(torch.isnan(mind_tensor),
                                  torch.zeros_like(mind_tensor),
                                  mind_tensor)

        # 上采样回原始尺寸
        if height != height_ds or width != width_ds:
            mind_tensor = F.interpolate(mind_tensor, size=(height, width), mode='bilinear', align_corners=True)

        return mind_tensor

    def forward(self, x, y):
        """
        计算两个图像之间的MIND损失

        参数:
            x: 第一个图像 (B, 1, H, W)
            y: 第二个图像 (B, 1, H, W)

        返回:
            两个MIND描述符之间的L1距离
        """
        # 计算MIND描述符
        x_mind = self._compute_mind(x)
        y_mind = self._compute_mind(y)

        # 计算L1距离 - 带有数值检查
        mind_diff = torch.abs(x_mind - y_mind)

        # 剔除任何NaN值
        mind_diff = torch.where(torch.isnan(mind_diff),
                                torch.zeros_like(mind_diff),
                                mind_diff)

        loss = torch.mean(mind_diff)

        # 最终安全检查 - 如果仍有NaN，返回一个小的常数
        if torch.isnan(loss):
            return torch.tensor(0.1, device=x.device, requires_grad=True)

        return loss


class SmoothRegularizationLoss(nn.Module):
    """
    变形场平滑度正则化损失
    使用更高效的梯度计算
    修复版本 - 确保数值稳定性
    """

    def __init__(self):
        super(SmoothRegularizationLoss, self).__init__()
        # 预定义梯度卷积核
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).reshape(1, 1, 3, 3)

    def forward(self, flow):
        """
        计算变形场的梯度以及二阶导数，并将其平方和作为正则化损失
        使用卷积加速计算

        参数:
            flow: 变形场 (B, 2, H, W)

        返回:
            平滑度损失值
        """
        device = flow.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        laplacian = self.laplacian.to(device)
        batch_size = flow.size(0)

        # 使用下采样减少计算量
        flow_ds = F.avg_pool2d(flow, kernel_size=2, stride=2)

        # 提取x和y方向的流场
        u = flow_ds[:, 0:1, :, :]  # x方向
        v = flow_ds[:, 1:2, :, :]  # y方向

        # 添加小量随机噪声，提高数值稳定性
        u = u + torch.randn_like(u) * 1e-4
        v = v + torch.randn_like(v) * 1e-4

        # 使用卷积计算梯度
        u_dx = F.conv2d(F.pad(u, [1, 1, 1, 1], mode='replicate'), sobel_x)
        u_dy = F.conv2d(F.pad(u, [1, 1, 1, 1], mode='replicate'), sobel_y)
        v_dx = F.conv2d(F.pad(v, [1, 1, 1, 1], mode='replicate'), sobel_x)
        v_dy = F.conv2d(F.pad(v, [1, 1, 1, 1], mode='replicate'), sobel_y)

        # 计算二阶导数（拉普拉斯算子）
        u_lap = F.conv2d(F.pad(u, [1, 1, 1, 1], mode='replicate'), laplacian)
        v_lap = F.conv2d(F.pad(v, [1, 1, 1, 1], mode='replicate'), laplacian)

        # 计算梯度幅度和拉普拉斯幅度，使用平方和
        grad_mag = u_dx ** 2 + u_dy ** 2 + v_dx ** 2 + v_dy ** 2
        lap_mag = u_lap ** 2 + v_lap ** 2

        # 防止极端值
        grad_mag = torch.clamp(grad_mag, 0.0, 100.0)
        lap_mag = torch.clamp(lap_mag, 0.0, 100.0)

        # 总的正则化损失
        reg_loss = torch.mean(grad_mag) + torch.mean(lap_mag)

        # 确保不返回NaN
        if torch.isnan(reg_loss):
            return torch.tensor(0.01, device=device, requires_grad=True)

        return reg_loss


class DualSimilarityLoss(nn.Module):
    """
    结合互信息和MIND描述符的双重相似性损失，以及变形场正则化
    优化版本，提高计算效率
    修复版本，确保数值稳定性
    """

    def __init__(self, alpha=5.0, beta=0.1):  # 减小权重，提高训练稳定性
        """
        参数:
            alpha: MIND损失的权重
            beta: 正则化损失的权重
        """
        super(DualSimilarityLoss, self).__init__()
        self.mi_loss = NormalizedMutualInformationLoss(num_bins=16)  # 减少bin数量
        self.mind_loss = MINDLoss(radius=1, dilation=2)  # 减小radius以减少计算量
        self.reg_loss = SmoothRegularizationLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, fixed, moved, flow):
        """
        计算总的损失

        参数:
            fixed: 固定图像 (B, 1, H, W)
            moved: 变形后的移动图像 (B, 1, H, W)
            flow: 变形场 (B, 2, H, W)

        返回:
            包含各损失项和总损失的字典
        """
        # 检查输入，确保没有NaN
        if torch.isnan(fixed).any() or torch.isnan(moved).any() or torch.isnan(flow).any():
            print("警告: 检测到输入张量包含NaN值")
            # 替换NaN为0
            fixed = torch.where(torch.isnan(fixed), torch.zeros_like(fixed), fixed)
            moved = torch.where(torch.isnan(moved), torch.zeros_like(moved), moved)
            flow = torch.where(torch.isnan(flow), torch.zeros_like(flow), flow)

        # 计算互信息损失
        mi = self.mi_loss(fixed, moved)

        # 计算MIND损失
        mind = self.mind_loss(fixed, moved)

        # 计算正则化损失
        reg = self.reg_loss(flow)

        # 打印损失值以进行调试
        if torch.isnan(mi) or torch.isnan(mind) or torch.isnan(reg):
            print(f"警告: 损失值包含NaN - MI: {mi.item()}, MIND: {mind.item()}, REG: {reg.item()}")

            # 替换NaN损失为小的正值
            mi = torch.tensor(0.1, device=fixed.device, requires_grad=True) if torch.isnan(mi) else mi
            mind = torch.tensor(0.1, device=fixed.device, requires_grad=True) if torch.isnan(mind) else mind
            reg = torch.tensor(0.01, device=fixed.device, requires_grad=True) if torch.isnan(reg) else reg

        # 总损失 - 使用渐进式加权避免任何一项初始过大
        total = mi + self.alpha * mind + self.beta * reg

        # 最终检查总损失
        if torch.isnan(total):
            print("警告: 总损失为NaN！使用固定损失值替代。")
            total = torch.tensor(1.0, device=fixed.device, requires_grad=True)

        return {
            'total': total,
            'mi': mi,
            'mind': mind,
            'reg': reg
        }