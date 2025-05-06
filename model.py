import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    卷积块，包含两个卷积层和批归一化层
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class DownsampleBlock(nn.Module):
    """
    下采样块，使用步长卷积
    """

    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.conv_block(x)
        return x


class UpsampleBlock(nn.Module):
    """
    上采样块
    """

    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 确保尺寸匹配（可能因为下采样/上采样导致尺寸不一致）
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])

        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_block(x)
        return x


class FlowEstimator(nn.Module):
    """
    变形场（光流）估计器
    """

    def __init__(self, in_channels):
        super(FlowEstimator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 2, kernel_size=3, padding=1)  # 输出2通道(x,y)位移场

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        flow = self.conv3(x)
        return flow


class SpatialTransformer(nn.Module):
    """
    空间变换器，用于对图像进行变形
    """

    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.grid_cache = {}  # 缓存不同尺寸的网格

    def _get_base_grid(self, B, H, W, device):
        """获取基础网格，带缓存以避免重复计算"""
        key = f"{H}_{W}_{device}"
        if key not in self.grid_cache:
            # 创建网格
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=2)
            # 不预先重复批次维度，而是在需要时动态扩展
            self.grid_cache[key] = grid.unsqueeze(0)

        # 动态扩展到当前批次大小
        base_grid = self.grid_cache[key]
        if base_grid.size(0) != B:
            base_grid = base_grid.repeat(B, 1, 1, 1)

        return base_grid

    def forward(self, img, flow):
        """
        参数:
            img: 输入图像 (B, C, H, W)
            flow: 变形场 (B, 2, H, W)

        返回:
            变形后的图像 (B, C, H, W)
        """
        # 图像尺寸
        B, C, H, W = img.size()
        flow_B = flow.size(0)
        device = img.device

        # 确保批次大小匹配
        if B != flow_B:
            # 如果批次大小不匹配，打印警告
            print(f"警告: 图像批次大小 ({B}) 与流场批次大小 ({flow_B}) 不匹配")
            # 如果flow的批次更小，则重复到img的批次大小
            if flow_B < B:
                flow = flow.repeat(B // flow_B, 1, 1, 1)
            # 如果img的批次更小，则只使用flow的前B个批次
            else:
                flow = flow[:B]

        # 调整流场尺寸
        flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # 归一化流场到[-1,1]
        flow = flow * 2.0 / torch.tensor([W - 1, H - 1], device=device)

        # 获取基础网格 - 确保批次大小正确
        grid = self._get_base_grid(B, H, W, device)

        # 应用流场变形
        grid = grid + flow

        # 使用grid_sample进行采样
        warped_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped_img


class MultiResolutionRegNet(nn.Module):
    """
    多分辨率配准网络
    """

    def __init__(self, in_channels=2):  # 修改为默认2通道 (灰度CT和灰度MRI)
        super(MultiResolutionRegNet, self).__init__()

        # 减少网络通道数
        # 编码器
        self.input_conv = ConvBlock(in_channels, 16)  # 从32减少到16
        self.down1 = DownsampleBlock(16, 32)  # 从64减少到32
        self.down2 = DownsampleBlock(32, 64)  # 从128减少到64
        self.down3 = DownsampleBlock(64, 128)  # 从256减少到128

        # 瓶颈层
        self.bottleneck = ConvBlock(128, 256)  # 从512减少到256

        # 解码器
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)

        # 多尺度输出 - 确保输入通道数正确
        self.flow_conv_lvl2 = FlowEstimator(64)  # 减少通道数
        self.flow_conv_lvl1 = FlowEstimator(32)
        self.flow_conv_lvl0 = FlowEstimator(16)

        # 空间变换器
        self.transformer = SpatialTransformer()

        # 上采样器（用于上采样流场）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, fixed, moving):
        """
        前向传播

        参数:
            fixed: 固定图像 (B, 1, H, W)
            moving: 移动图像 (B, 1, H, W)

        返回:
            字典，包含多尺度的变形结果和流场
        """
        # 确保输入尺寸匹配
        if fixed.size() != moving.size():
            print(f"警告: 输入尺寸不匹配 - fixed: {fixed.shape}, moving: {moving.shape}")
            # 如果尺寸不匹配，将moving调整为与fixed相同的尺寸
            if fixed.size(0) != moving.size(0):
                if fixed.size(0) > moving.size(0):
                    # 重复moving到fixed的批次大小
                    repeat_factor = fixed.size(0) // moving.size(0)
                    moving = moving.repeat(repeat_factor, 1, 1, 1)
                else:
                    # 截取fixed大小的batch
                    moving = moving[:fixed.size(0)]

        # 拼接输入图像
        x = torch.cat([fixed, moving], dim=1)  # (B, 2, H, W)

        # 编码器
        x1 = self.input_conv(x)  # Level 0特征
        x2 = self.down1(x1)  # Level 1特征
        x3 = self.down2(x2)  # Level 2特征
        x4 = self.down3(x3)  # Level 3特征

        # 瓶颈层
        x5 = self.bottleneck(x4)

        # 解码器
        x = self.up1(x5, x4)

        # Level 2分辨率的流场
        x = self.up2(x, x3)
        flow_lvl2 = self.flow_conv_lvl2(x)

        # 确保在正确的尺寸上应用变换
        # 调整moving图像到flow_lvl2的尺寸
        moving_resized = F.interpolate(moving, size=flow_lvl2.shape[2:], mode='bilinear', align_corners=True)
        warped_lvl2 = self.transformer(moving_resized, flow_lvl2)

        # 上采样流场
        flow_up_lvl2 = self.upsample(flow_lvl2) * 2.0  # 尺度因子

        # Level 1分辨率的流场
        x = self.up3(x, x2)
        flow_lvl1 = self.flow_conv_lvl1(x) + flow_up_lvl2

        # 确保在正确的尺寸上应用变换
        # 调整moving图像到flow_lvl1的尺寸
        moving_resized = F.interpolate(moving, size=flow_lvl1.shape[2:], mode='bilinear', align_corners=True)
        warped_lvl1 = self.transformer(moving_resized, flow_lvl1)

        # 上采样流场
        flow_up_lvl1 = self.upsample(flow_lvl1) * 2.0  # 尺度因子

        # Level 0（原始分辨率）的流场
        flow_lvl0 = self.flow_conv_lvl0(x1) + flow_up_lvl1

        warped_lvl0 = self.transformer(moving, flow_lvl0)

        return {
            'warped_lvl0': warped_lvl0,
            'warped_lvl1': warped_lvl1,
            'warped_lvl2': warped_lvl2,
            'flow_lvl0': flow_lvl0,
            'flow_lvl1': flow_lvl1,
            'flow_lvl2': flow_lvl2
        }