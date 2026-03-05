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


# ====================== 创新点1：基于Mamba的选择性状态空间模块 ======================
class SelectiveSSM(nn.Module):
    """
    选择性状态空间模块 (Selective State Space Module)

    借鉴Mamba的思想，实现能够选择性地捕获序列中的重要信息，并根据内容自适应地处理信息
    对于医学图像配准任务，这有助于网络聚焦在关键解剖结构上，同时忽略不相关的信息
    """

    def __init__(self, channels, reduction=4, dropout=0.1):
        super(SelectiveSSM, self).__init__()
        # 简化实现，避免序列处理的高计算开销
        self.channels = channels
        self.reduced_dim = max(16, channels // reduction)

        # 将2D特征转换为序列处理的更高效实现
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU()
        )

        # 使用深度可分离卷积模拟SSM的特性
        self.depth_conv = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels
        )

        # 模拟选择性状态更新的门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([channels, 1, 1])

    def forward(self, x):
        """
        输入：x [B, C, H, W]
        输出：y [B, C, H, W]
        """
        residual = x

        # 通道混合
        x_mixed = self.channel_mixer(x)

        # 深度卷积模拟序列处理
        x_seq = self.depth_conv(x_mixed)

        # 门控机制 - 模拟选择性状态更新
        gate_value = self.gate(x_seq)
        x_gated = x_seq * gate_value

        # 输出投影
        output = self.output_proj(x_gated)

        # 残差连接
        output = output + residual

        return output


# 简化版选择性SSM模块，计算效率更高
class EfficientSSM(nn.Module):
    """
    高效的选择性状态空间模块，简化计算同时保持选择性特性
    """

    def __init__(self, channels, reduction=8, kernel_size=3):
        super(EfficientSSM, self).__init__()
        self.channels = channels
        self.reduced_channels = max(8, channels // reduction)

        # 特征降维
        self.reduce = nn.Conv2d(channels, self.reduced_channels, kernel_size=1)

        # 用卷积近似SSM操作
        self.conv_x = nn.Conv2d(self.reduced_channels, self.reduced_channels,
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_h = nn.Conv2d(self.reduced_channels, self.reduced_channels,
                                kernel_size=kernel_size, padding=kernel_size // 2)

        # 状态门控 - 类似LSTM的门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(self.reduced_channels * 2, self.reduced_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征恢复
        self.expand = nn.Conv2d(self.reduced_channels, channels, kernel_size=1)

        # 激活函数
        self.activation = nn.SiLU()

    def forward(self, x):
        identity = x

        # 特征降维
        x_reduced = self.reduce(x)

        # 状态更新
        x_h = self.conv_h(x_reduced)
        x_x = self.conv_x(x_reduced)

        # 计算门控系数，决定保留多少历史信息
        gate = self.gate(torch.cat([x_h, x_x], dim=1))

        # 应用选择性状态更新
        out = x_h * gate + x_x * (1 - gate)

        # 特征恢复
        out = self.expand(out)

        # 残差连接
        return self.activation(out + identity)


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

        # 在上采样块中集成高效的SSM模块，提高特征聚合能力
        self.ssm = EfficientSSM(out_channels)

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

        # 应用选择性SSM增强特征表达
        x = self.ssm(x)

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
        # MPS 不支持 'border'，改用 'zeros'（医学图像边界外填充黑色背景合理）
        padding_mode = 'zeros' if device.type == 'mps' else 'border'
        warped_img = F.grid_sample(img, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)

        return warped_img


# ====================== 创新点2：基于Titans的长期记忆机制 ======================
class LongTermMemory(nn.Module):
    """
    长期记忆模块 (Long-Term Memory Module)

    借鉴Titans的思想，实现一个能够在配准过程中学习和记忆关键特征的模块
    这有助于网络持续关注重要的解剖结构，提高配准准确性，尤其是对于难以配准的区域
    """

    def __init__(self, channels, memory_size=16, update_rate=0.1):
        super(LongTermMemory, self).__init__()
        self.channels = channels
        self.memory_size = memory_size
        self.update_rate = update_rate

        # 简化内存表示，使用卷积形式实现以避免维度不匹配问题
        self.memory_feature = nn.Parameter(torch.randn(1, memory_size, 1, 1))
        self.memory_key = nn.Parameter(torch.randn(1, channels, memory_size))

        # 特征变换和压缩
        self.feature_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU()
        )

        # 输出增强
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        # 更新门控
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 初始化参数
        nn.init.normal_(self.memory_feature, 0.0, 0.02)
        nn.init.normal_(self.memory_key, 0.0, 0.02)

    def forward(self, x):
        """
        输入：x [B, C, H, W]
        输出：enhanced_x [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x

        # 特征投影
        x_proj = self.feature_proj(x)

        # 提取全局上下文特征
        x_global = self.compress(x_proj)  # [B, C, 1, 1]

        # 计算当前特征与记忆的相关性
        # 将memory_key扩展到批次维度
        expanded_memory_key = self.memory_key.expand(B, -1, -1)  # [B, C, M]

        # 将全局特征与记忆键计算相关性
        # x_global: [B, C, 1, 1] -> [B, C]
        # expanded_memory_key: [B, C, M]
        x_global_flat = x_global.squeeze(-1).squeeze(-1)  # [B, C]

        # 计算注意力得分: [B, C] @ [B, C, M] -> [B, M]
        attn_scores = torch.bmm(
            x_global_flat.unsqueeze(1),  # [B, 1, C]
            expanded_memory_key  # [B, C, M]
        ).squeeze(1)  # [B, M]

        # 应用softmax得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, M]

        # 获取加权的记忆特征
        # 扩展memory_feature到批次维度
        expanded_memory = self.memory_feature.expand(B, -1, 1, 1)  # [B, M, 1, 1]

        # 使用注意力权重加权求和
        # attn_weights: [B, M] -> [B, M, 1, 1]
        # expanded_memory: [B, M, 1, 1]
        weighted_memory = (
                expanded_memory * attn_weights.view(B, self.memory_size, 1, 1)
        ).sum(dim=1, keepdim=True)  # [B, 1, 1, 1]

        # 增强输入特征
        memory_enhanced = weighted_memory * x_proj

        # 应用增强层
        enhanced_features = self.enhance(memory_enhanced)

        # 计算门控系数并应用
        gate_value = self.gate(x_proj)
        output = identity + gate_value * enhanced_features

        # 训练时更新记忆（简化实现，使用滑动平均）
        if self.training:
            # 提取当前批次最显著特征
            x_significant = x_global  # [B, C, 1, 1]

            # 计算当前特征与记忆之间的"惊喜度"
            surprise_factor = 1.0 - F.cosine_similarity(
                x_global_flat,  # [B, C]
                torch.mean(expanded_memory_key, dim=-1),  # [B, C]
                dim=1
            ).view(B, 1, 1, 1)  # [B, 1, 1, 1]

            # 基于惊喜度更新记忆
            with torch.no_grad():
                # 对惊喜度进行排序
                _, indices = torch.sort(surprise_factor.view(-1), descending=True)
                update_indices = indices[:min(B, self.memory_size // 4)]

                # 更新记忆特征和键（只使用部分最惊喜的样本）
                if len(update_indices) > 0:
                    for i, batch_idx in enumerate(update_indices):
                        # 随机选择要更新的记忆位置
                        mem_idx = torch.randint(0, self.memory_size, (1,)).item()

                        # 提取要更新的特征和键
                        selected_batch = batch_idx.item() % B

                        # 动量更新记忆特征 - 直接访问数据并确保形状匹配
                        old_mem_feature = self.memory_feature.data[0, mem_idx].clone()
                        new_mem_feature = x_significant[selected_batch, :, 0, 0].mean().item()

                        # 更新为标量值
                        self.memory_feature.data[0, mem_idx] = (
                                (1 - self.update_rate) * old_mem_feature +
                                self.update_rate * new_mem_feature
                        )

                        # 更新对应的键
                        old_key = self.memory_key.data[0, :, mem_idx].clone()
                        new_key = x_global_flat[selected_batch]

                        self.memory_key.data[0, :, mem_idx] = (
                                (1 - self.update_rate) * old_key +
                                self.update_rate * new_key
                        )

        return output


class MultiResolutionRegNet(nn.Module):
    """
    多分辨率配准网络
    """

    def __init__(self, in_channels=2):  # 修改为默认2通道 (灰度CT和灰度MRI)
        super(MultiResolutionRegNet, self).__init__()

        # 编码器
        self.input_conv = ConvBlock(in_channels, 16)
        self.down1 = DownsampleBlock(16, 32)
        self.down2 = DownsampleBlock(32, 64)
        self.down3 = DownsampleBlock(64, 128)

        # 瓶颈层
        self.bottleneck = ConvBlock(128, 256)

        # 创新点1: 在瓶颈层应用选择性SSM
        self.bottleneck_ssm = SelectiveSSM(256)

        # 解码器
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)

        # 创新点2: 长期记忆模块集成到每个解码层
        self.memory_module1 = LongTermMemory(128)
        self.memory_module2 = LongTermMemory(64)
        self.memory_module3 = LongTermMemory(32)

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

        # 创新点1: 应用选择性SSM增强瓶颈层特征
        x5 = self.bottleneck_ssm(x5)

        # 解码器
        x = self.up1(x5, x4)

        # 创新点2: 应用长期记忆增强特征
        x = self.memory_module1(x)

        # Level 2分辨率的流场
        x = self.up2(x, x3)

        # 创新点2: 应用长期记忆增强特征
        x = self.memory_module2(x)

        flow_lvl2 = self.flow_conv_lvl2(x)

        # 确保在正确的尺寸上应用变换
        # 调整moving图像到flow_lvl2的尺寸
        moving_resized = F.interpolate(moving, size=flow_lvl2.shape[2:], mode='bilinear', align_corners=True)
        warped_lvl2 = self.transformer(moving_resized, flow_lvl2)

        # 上采样流场
        flow_up_lvl2 = self.upsample(flow_lvl2) * 2.0  # 尺度因子

        # Level 1分辨率的流场
        x = self.up3(x, x2)

        # 创新点2: 应用长期记忆增强特征
        x = self.memory_module3(x)

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