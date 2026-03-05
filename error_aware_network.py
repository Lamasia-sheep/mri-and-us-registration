"""
3.2.3 误差感知门控机制
3.2.4 MRI解剖约束修正机制

将误差先验信息嵌入SelectiveSSM模块，实现误差感知的配准修正
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConfidenceMapGenerator(nn.Module):
    """
    置信度图生成器
    
    将三维误差分布转换为二维超声图像平面上的置信度图
    支持动态更新以适应探头位置变化
    """
    
    def __init__(self, image_size=(256, 256), tau=0.949):
        """
        参数:
            image_size: 超声图像尺寸 (H, W)
            tau: 温度参数，控制置信度对误差的敏感程度
        """
        super(ConfidenceMapGenerator, self).__init__()
        self.image_size = image_size
        self.tau = tau
        
        # 预计算基础置信度图 (可以从误差先验模型加载)
        # 这里使用一个可学习的参数作为示例
        self.register_buffer('base_confidence', 
                            torch.ones(1, 1, *image_size) * 0.5)
        
    def forward(self, sigma_2d=None):
        """
        计算置信度图
        
        参数:
            sigma_2d: (H, W) 2D误差标准差图，如果为None则使用预设值
        返回:
            confidence_map: (1, 1, H, W) 置信度图
        """
        if sigma_2d is None:
            return self.base_confidence
        
        # C(u,v) = exp(-σ²_2D(u,v) / 2τ²)
        confidence = torch.exp(-sigma_2d**2 / (2 * self.tau**2))
        
        if confidence.dim() == 2:
            confidence = confidence.unsqueeze(0).unsqueeze(0)
        
        return confidence
    
    def load_from_error_model(self, confidence_map_np):
        """
        从误差先验模型加载置信度图
        
        参数:
            confidence_map_np: numpy数组格式的置信度图
        """
        confidence_tensor = torch.from_numpy(confidence_map_np).float()
        if confidence_tensor.dim() == 2:
            confidence_tensor = confidence_tensor.unsqueeze(0).unsqueeze(0)
        self.base_confidence = confidence_tensor


class ErrorAwareSelectiveSSM(nn.Module):
    """
    误差感知选择性状态空间模块
    
    核心思想：让网络"知道"哪些区域的输入坐标是可信的
    - 高置信区域：网络可以充分信任输入特征，正常进行特征选择和变换
    - 低置信区域：网络应该保持保守，减少对可能错误特征的依赖
    
    实现方式：将置信度图嵌入SelectiveSSM模块的门控机制
    """
    
    def __init__(self, channels, reduction=4, dropout=0.1):
        super(ErrorAwareSelectiveSSM, self).__init__()
        self.channels = channels
        self.reduced_dim = max(16, channels // reduction)
        
        # 通道混合
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU()
        )
        
        # 深度可分离卷积模拟SSM的特性
        self.depth_conv = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels
        )
        
        # 原始门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, confidence_map=None):
        """
        误差感知的前向传播
        
        参数:
            x: 输入特征 [B, C, H, W]
            confidence_map: 置信度图 [B, 1, H, W] 或 [1, 1, H_orig, W_orig]
        返回:
            y: 输出特征 [B, C, H, W]
        """
        residual = x
        B, C, H, W = x.shape
        
        # 通道混合
        x_mixed = self.channel_mixer(x)
        
        # 深度卷积模拟序列处理
        x_seq = self.depth_conv(x_mixed)
        
        # 计算原始门控信号 G
        G = self.gate(x_seq)
        
        # 误差感知门控
        if confidence_map is not None:
            # 对齐置信度图到当前特征尺寸
            C_aligned = self._align_confidence_map(confidence_map, H, W)
            
            # 误差感知门控: G_EA = G ⊙ C_aligned
            G_EA = G * C_aligned
        else:
            G_EA = G
        
        # 特征更新: F_out = G_EA ⊙ DepthConv(F_in) + (1 - G_EA) ⊙ F_in
        x_gated = G_EA * x_seq + (1 - G_EA) * x_mixed
        
        # 输出投影
        output = self.output_proj(x_gated)
        
        # 残差连接
        output = output + residual
        
        return output
    
    def _align_confidence_map(self, confidence_map, target_h, target_w):
        """
        对齐置信度图到目标特征尺寸
        
        由于特征图经过下采样，尺寸小于原始图像，
        需要对置信度图进行下采样对齐
        
        参数:
            confidence_map: 原始置信度图 [B, 1, H_orig, W_orig]
            target_h, target_w: 目标尺寸
        返回:
            C_aligned: 对齐后的置信度图 [B, 1, target_h, target_w]
        """
        if confidence_map.shape[2] != target_h or confidence_map.shape[3] != target_w:
            C_aligned = F.adaptive_avg_pool2d(confidence_map, (target_h, target_w))
        else:
            C_aligned = confidence_map
        
        return C_aligned


class MRIAnatomicalConstraint(nn.Module):
    """
    MRI解剖约束修正机制
    
    利用LTM模块存储的MRI解剖特征，为低置信区域提供可靠的参考信息
    
    修正触发条件: C(u,v) < C_th (置信度阈值)
    """
    
    def __init__(self, channels, memory_size=16, confidence_threshold=0.4, 
                 max_correction_strength=0.5):
        """
        参数:
            channels: 特征通道数
            memory_size: 记忆槽数量
            confidence_threshold: 置信度阈值，低于此值触发修正
            max_correction_strength: 最大修正强度γ
        """
        super(MRIAnatomicalConstraint, self).__init__()
        self.channels = channels
        self.memory_size = memory_size
        self.confidence_threshold = confidence_threshold
        self.gamma = max_correction_strength  # 最大修正强度
        
        # 记忆特征和记忆键
        self.memory_feature = nn.Parameter(torch.randn(1, memory_size, channels))
        self.memory_key = nn.Parameter(torch.randn(1, channels, memory_size))
        
        # 特征投影
        self.query_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(channels, channels)
        )
        
        # 初始化
        nn.init.normal_(self.memory_feature, 0.0, 0.02)
        nn.init.normal_(self.memory_key, 0.0, 0.02)
        
    def forward(self, x, confidence_map):
        """
        MRI约束修正
        
        参数:
            x: 输入特征 [B, C, H, W] (通常是超声特征)
            confidence_map: 置信度图 [B, 1, H, W]
        返回:
            x_corrected: 修正后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 对齐置信度图
        if confidence_map.shape[2] != H or confidence_map.shape[3] != W:
            confidence_map = F.adaptive_avg_pool2d(confidence_map, (H, W))
        
        # 步骤1: MRI记忆特征检索
        # q = GlobalAvgPool(F_US)
        q = self.query_proj(x)  # [B, C]
        
        # 计算注意力分数: Scores = Softmax(q · MemoryKey^T / sqrt(d))
        # memory_key: [1, C, M] -> [B, C, M]
        memory_key = self.memory_key.expand(B, -1, -1)
        
        # [B, 1, C] @ [B, C, M] -> [B, 1, M] -> [B, M]
        scores = torch.bmm(q.unsqueeze(1), memory_key).squeeze(1)
        scores = scores / (C ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [B, M]
        
        # 检索MRI记忆特征: F_MRI_mem = Scores · MemoryFeature
        # memory_feature: [1, M, C] -> [B, M, C]
        memory_feature = self.memory_feature.expand(B, -1, -1)
        
        # [B, 1, M] @ [B, M, C] -> [B, 1, C] -> [B, C]
        F_MRI_mem = torch.bmm(attn_weights.unsqueeze(1), memory_feature).squeeze(1)
        
        # 扩展到空间维度 [B, C, 1, 1] -> [B, C, H, W]
        F_MRI_mem = F_MRI_mem.view(B, C, 1, 1).expand(-1, -1, H, W)
        
        # 步骤2: 计算修正强度λ
        # λ(u,v) = 0 if C(u,v) >= C_th
        #        = γ * (1 - C(u,v)/C_th) if C(u,v) < C_th
        trigger_mask = (confidence_map < self.confidence_threshold).float()
        lambda_map = self.gamma * (1 - confidence_map / self.confidence_threshold) * trigger_mask
        
        # 步骤3: 自适应特征修正
        # F_corrected(u,v) = F_US(u,v) + λ(u,v) * (F_MRI_mem(u,v) - F_US(u,v))
        x_corrected = x + lambda_map * (F_MRI_mem - x)
        
        return x_corrected, lambda_map


class ErrorAwareRegistrationModule(nn.Module):
    """
    误差感知配准修正模块
    
    整合误差感知门控机制和MRI解剖约束修正
    """
    
    def __init__(self, channels, memory_size=16, confidence_threshold=0.4):
        super(ErrorAwareRegistrationModule, self).__init__()
        
        self.error_aware_ssm = ErrorAwareSelectiveSSM(channels)
        self.mri_constraint = MRIAnatomicalConstraint(
            channels, memory_size, confidence_threshold
        )
        
    def forward(self, x, confidence_map):
        """
        参数:
            x: 输入特征 [B, C, H, W]
            confidence_map: 置信度图 [B, 1, H, W]
        返回:
            x_out: 修正后的特征
            correction_info: 包含修正信息的字典
        """
        # 误差感知门控
        x_gated = self.error_aware_ssm(x, confidence_map)
        
        # MRI解剖约束修正
        x_corrected, lambda_map = self.mri_constraint(x_gated, confidence_map)
        
        correction_info = {
            'lambda_map': lambda_map,
            'gated_features': x_gated
        }
        
        return x_corrected, correction_info


class MultiScaleConfidenceEmbedding(nn.Module):
    """
    多尺度置信度嵌入
    
    由于SelectiveSSM模块在编码器的多个层级都有使用，
    需要在每个层级嵌入对应尺度的置信度图
    """
    
    def __init__(self, num_scales=4):
        super(MultiScaleConfidenceEmbedding, self).__init__()
        self.num_scales = num_scales
        
    def forward(self, confidence_map, scale_factors):
        """
        生成多尺度置信度图
        
        参数:
            confidence_map: 原始置信度图 [B, 1, H, W]
            scale_factors: 各尺度的下采样因子列表, e.g., [1, 2, 4, 8]
        返回:
            multi_scale_confidence: 多尺度置信度图列表
        """
        multi_scale_confidence = []
        
        for sf in scale_factors:
            if sf == 1:
                C_scaled = confidence_map
            else:
                target_size = (confidence_map.shape[2] // sf, 
                              confidence_map.shape[3] // sf)
                C_scaled = F.adaptive_avg_pool2d(confidence_map, target_size)
            
            multi_scale_confidence.append(C_scaled)
        
        return multi_scale_confidence


class SmoothCorrectionLoss(nn.Module):
    """
    修正机制的物理约束损失
    
    包括:
    1. 空间一致性约束：修正向量应该在空间上平滑变化
    2. 幅度约束：单像素修正幅度不应过大
    """
    
    def __init__(self, max_correction_magnitude=0.1):
        super(SmoothCorrectionLoss, self).__init__()
        self.delta_max = max_correction_magnitude
        
    def forward(self, lambda_map, x_original, x_corrected):
        """
        计算修正约束损失
        
        参数:
            lambda_map: 修正强度图 [B, 1, H, W]
            x_original: 原始特征 [B, C, H, W]
            x_corrected: 修正后特征 [B, C, H, W]
        返回:
            total_loss: 总约束损失
            loss_dict: 各项损失的字典
        """
        # 1. 空间一致性约束: L_smooth = Σ ||∇λ(u,v)||²
        # 计算λ的梯度
        lambda_dx = lambda_map[:, :, :, 1:] - lambda_map[:, :, :, :-1]
        lambda_dy = lambda_map[:, :, 1:, :] - lambda_map[:, :, :-1, :]
        
        smooth_loss = torch.mean(lambda_dx**2) + torch.mean(lambda_dy**2)
        
        # 2. 幅度约束: ||F_corrected - F_original||_2 ≤ δ_max
        correction_magnitude = torch.norm(x_corrected - x_original, dim=1, keepdim=True)
        
        # 使用软约束
        magnitude_loss = F.relu(correction_magnitude - self.delta_max).mean()
        
        total_loss = smooth_loss + magnitude_loss
        
        loss_dict = {
            'smooth_loss': smooth_loss,
            'magnitude_loss': magnitude_loss
        }
        
        return total_loss, loss_dict


# ===================== 集成到完整网络 =====================

class ErrorAwareMultiResolutionRegNet(nn.Module):
    """
    误差感知的多分辨率配准网络
    
    基于第二章的SelectiveSSM-LTMNet，集成:
    - 3.2.2 误差先验建模（置信度图）
    - 3.2.3 误差感知门控机制
    - 3.2.4 MRI解剖约束修正
    """
    
    def __init__(self, in_channels=2, confidence_threshold=0.4):
        super(ErrorAwareMultiResolutionRegNet, self).__init__()
        
        # 编码器 (与原网络相同)
        self.input_conv = ConvBlock(in_channels, 16)
        self.down1 = DownsampleBlock(16, 32)
        self.down2 = DownsampleBlock(32, 64)
        self.down3 = DownsampleBlock(64, 128)
        
        # 瓶颈层 - 误差感知SelectiveSSM
        self.bottleneck = ConvBlock(128, 256)
        self.bottleneck_ea_ssm = ErrorAwareSelectiveSSM(256)
        
        # 解码器 - 带误差感知的上采样
        self.up1 = ErrorAwareUpsampleBlock(256, 128, confidence_threshold)
        self.up2 = ErrorAwareUpsampleBlock(128, 64, confidence_threshold)
        self.up3 = ErrorAwareUpsampleBlock(64, 32, confidence_threshold)
        
        # MRI约束修正模块
        self.mri_constraint = MRIAnatomicalConstraint(
            channels=256, memory_size=16, 
            confidence_threshold=confidence_threshold
        )
        
        # 多尺度流场估计
        self.flow_conv_lvl2 = FlowEstimator(64)
        self.flow_conv_lvl1 = FlowEstimator(32)
        self.flow_conv_lvl0 = FlowEstimator(16)
        
        # 空间变换器
        self.transformer = SpatialTransformer()
        
        # 多尺度置信度嵌入
        self.ms_confidence = MultiScaleConfidenceEmbedding(num_scales=4)
        
        # 上采样器
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, fixed, moving, confidence_map=None):
        """
        前向传播
        
        参数:
            fixed: 固定图像 (B, 1, H, W)
            moving: 移动图像 (B, 1, H, W)
            confidence_map: 置信度图 (B, 1, H, W)，如果为None则不使用误差感知
        返回:
            outputs: 包含配准结果和中间特征的字典
        """
        # 生成多尺度置信度图
        if confidence_map is not None:
            scale_factors = [1, 2, 4, 8]
            ms_conf = self.ms_confidence(confidence_map, scale_factors)
            conf_s1, conf_s2, conf_s3, conf_s4 = ms_conf
        else:
            conf_s1 = conf_s2 = conf_s3 = conf_s4 = None
        
        # 拼接输入
        x = torch.cat([fixed, moving], dim=1)
        
        # 编码器
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 瓶颈层 + 误差感知SSM
        x5 = self.bottleneck(x4)
        x5 = self.bottleneck_ea_ssm(x5, conf_s4)
        
        # MRI约束修正 (在瓶颈层应用)
        if confidence_map is not None:
            x5_corrected, correction_info = self.mri_constraint(x5, conf_s4)
        else:
            x5_corrected = x5
            correction_info = None
        
        # 解码器
        x = self.up1(x5_corrected, x4, conf_s3)
        x = self.up2(x, x3, conf_s2)
        
        # Level 2流场
        flow_lvl2 = self.flow_conv_lvl2(x)
        moving_resized = F.interpolate(moving, size=flow_lvl2.shape[2:], 
                                       mode='bilinear', align_corners=True)
        warped_lvl2 = self.transformer(moving_resized, flow_lvl2)
        
        # 上采样流场
        flow_up_lvl2 = self.upsample(flow_lvl2) * 2.0
        
        # Level 1
        x = self.up3(x, x2, conf_s1)
        flow_lvl1 = self.flow_conv_lvl1(x) + flow_up_lvl2
        moving_resized = F.interpolate(moving, size=flow_lvl1.shape[2:],
                                       mode='bilinear', align_corners=True)
        warped_lvl1 = self.transformer(moving_resized, flow_lvl1)
        
        # 上采样流场
        flow_up_lvl1 = self.upsample(flow_lvl1) * 2.0
        
        # Level 0
        flow_lvl0 = self.flow_conv_lvl0(x1) + flow_up_lvl1
        warped_lvl0 = self.transformer(moving, flow_lvl0)
        
        outputs = {
            'warped_lvl0': warped_lvl0,
            'warped_lvl1': warped_lvl1,
            'warped_lvl2': warped_lvl2,
            'flow_lvl0': flow_lvl0,
            'flow_lvl1': flow_lvl1,
            'flow_lvl2': flow_lvl2,
            'correction_info': correction_info
        }
        
        return outputs


# ===================== 辅助模块定义 =====================

class ConvBlock(nn.Module):
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


class ErrorAwareUpsampleBlock(nn.Module):
    """带误差感知的上采样模块"""
    def __init__(self, in_channels, out_channels, confidence_threshold=0.4):
        super(ErrorAwareUpsampleBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.ea_ssm = ErrorAwareSelectiveSSM(out_channels)

    def forward(self, x1, x2, confidence_map=None):
        x1 = self.up(x1)
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_block(x)
        x = self.ea_ssm(x, confidence_map)
        return x


class FlowEstimator(nn.Module):
    def __init__(self, in_channels):
        super(FlowEstimator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        flow = self.conv3(x)
        return flow


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.grid_cache = {}

    def forward(self, img, flow):
        B, C, H, W = img.size()
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
        flow_perm = flow.permute(0, 2, 3, 1)
        flow_norm = flow_perm * 2.0 / torch.tensor([W - 1, H - 1], device=img.device)
        grid = grid + flow_norm
        warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped


if __name__ == "__main__":
    # 测试误差感知网络
    print("测试误差感知配准网络...")
    
    # 创建模型
    model = ErrorAwareMultiResolutionRegNet(in_channels=2, confidence_threshold=0.4)
    
    # 创建测试输入
    B, H, W = 2, 256, 256
    fixed = torch.randn(B, 1, H, W)
    moving = torch.randn(B, 1, H, W)
    confidence_map = torch.rand(B, 1, H, W) * 0.5 + 0.3  # [0.3, 0.8]
    
    # 前向传播
    outputs = model(fixed, moving, confidence_map)
    
    print(f"输入尺寸: fixed={fixed.shape}, moving={moving.shape}")
    print(f"置信度图尺寸: {confidence_map.shape}")
    print(f"输出warped_lvl0尺寸: {outputs['warped_lvl0'].shape}")
    print(f"输出flow_lvl0尺寸: {outputs['flow_lvl0'].shape}")
    
    if outputs['correction_info'] is not None:
        print(f"修正强度图尺寸: {outputs['correction_info']['lambda_map'].shape}")
    
    print("测试通过!")
        