"""
3.3 面向边缘部署的轻量化与抗噪优化

本模块实现:
- 3.3.1 轻量化需求分析
- 3.3.2 基于敏感度分析的结构化剪枝
- 3.3.3 误差感知知识蒸馏
- 3.3.4 动态噪声抑制（DNS）模块
- 3.3.5 边缘设备部署优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
from collections import OrderedDict


# ===================== 3.3.1 轻量化需求分析 =====================

class ModelProfiler:
    """
    网络计算量分析器
    
    对SelectiveSSM-LTMNet进行逐层计算量分析，
    确定优化方向和剪枝策略
    """
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 2, 256, 256)):
        """
        参数:
            model: 待分析的网络模型
            input_size: 输入张量尺寸 (B, C, H, W)
        """
        self.model = model
        self.input_size = input_size
        self.layer_info = {}
        self.total_flops = 0
        self.total_params = 0
        
    def count_conv2d_flops(self, module: nn.Conv2d, input_shape: Tuple, output_shape: Tuple) -> int:
        """计算Conv2d的FLOPs"""
        batch_size, in_channels, in_h, in_w = input_shape
        out_channels, _, out_h, out_w = output_shape
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (in_channels // module.groups)
        output_elements = out_channels * out_h * out_w
        
        # 乘加运算
        flops = 2 * kernel_ops * output_elements * batch_size
        
        # 偏置
        if module.bias is not None:
            flops += output_elements * batch_size
            
        return flops
    
    def count_linear_flops(self, module: nn.Linear, input_shape: Tuple) -> int:
        """计算Linear的FLOPs"""
        batch_size = input_shape[0]
        flops = 2 * module.in_features * module.out_features * batch_size
        if module.bias is not None:
            flops += module.out_features * batch_size
        return flops
    
    def profile(self) -> Dict:
        """
        执行网络分析
        
        返回:
            profile_results: 各层计算量分析结果
        """
        self.layer_info = {}
        self.total_flops = 0
        self.total_params = 0
        
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                # 参数量
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # FLOPs估计
                flops = 0
                if isinstance(module, nn.Conv2d):
                    if len(input) > 0 and input[0] is not None:
                        input_shape = tuple(input[0].shape)
                        output_shape = tuple(output.shape)
                        flops = self.count_conv2d_flops(module, input_shape, output_shape)
                elif isinstance(module, nn.Linear):
                    if len(input) > 0 and input[0] is not None:
                        input_shape = tuple(input[0].shape)
                        flops = self.count_linear_flops(module, input_shape)
                
                self.layer_info[name] = {
                    'type': module.__class__.__name__,
                    'params': params,
                    'flops': flops
                }
                self.total_flops += flops
                self.total_params += params
                
            return hook
        
        # 注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # 前向传播
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(self.input_size).to(device)
        
        # 根据模型输入参数调整
        with torch.no_grad():
            try:
                # 假设模型需要fixed和moving两个输入
                if self.input_size[1] == 2:
                    fixed = dummy_input[:, :1]
                    moving = dummy_input[:, 1:]
                    self.model(fixed, moving)
                else:
                    self.model(dummy_input)
            except Exception as e:
                print(f"分析时遇到问题: {e}")
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return {
            'layer_info': self.layer_info,
            'total_flops': self.total_flops,
            'total_params': self.total_params,
            'flops_giga': self.total_flops / 1e9,
            'params_mega': self.total_params / 1e6
        }
    
    def print_summary(self):
        """打印分析摘要"""
        print("\n" + "=" * 70)
        print("模型计算量分析摘要")
        print("=" * 70)
        print(f"总FLOPs: {self.total_flops / 1e9:.2f} G")
        print(f"总参数量: {self.total_params / 1e6:.2f} M")
        print("-" * 70)
        
        # 按模块类型分组统计
        type_stats = {}
        for name, info in self.layer_info.items():
            layer_type = info['type']
            if layer_type not in type_stats:
                type_stats[layer_type] = {'flops': 0, 'params': 0, 'count': 0}
            type_stats[layer_type]['flops'] += info['flops']
            type_stats[layer_type]['params'] += info['params']
            type_stats[layer_type]['count'] += 1
        
        print(f"{'层类型':<20} {'数量':>8} {'FLOPs(G)':>12} {'参数(M)':>12} {'占比':>8}")
        print("-" * 70)
        for layer_type, stats in sorted(type_stats.items(), key=lambda x: -x[1]['flops']):
            flops_g = stats['flops'] / 1e9
            params_m = stats['params'] / 1e6
            ratio = stats['flops'] / max(self.total_flops, 1) * 100
            print(f"{layer_type:<20} {stats['count']:>8} {flops_g:>12.3f} {params_m:>12.3f} {ratio:>7.1f}%")


# ===================== 3.3.2 基于敏感度分析的结构化剪枝 =====================

class SensitivityAnalyzer:
    """
    敏感度分析器
    
    对网络每一层进行敏感度评估，确定可剪枝程度
    敏感度定义: S_L = ΔDICE / Δpruning_ratio
    """
    
    def __init__(self, model: nn.Module, dataloader, criterion, device='cuda'):
        """
        参数:
            model: 原始网络模型
            dataloader: 验证数据加载器
            criterion: 损失函数/评估指标
            device: 计算设备
        """
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.sensitivity_scores = {}
        
    def _get_prunable_layers(self) -> Dict[str, nn.Conv2d]:
        """获取可剪枝的卷积层"""
        prunable = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 跳过深度可分离卷积的深度卷积部分
                if module.groups == 1 or module.groups != module.in_channels:
                    prunable[name] = module
        return prunable
    
    def _compute_l1_importance(self, conv: nn.Conv2d) -> torch.Tensor:
        """
        计算卷积核的L1范数重要性
        
        Importance_c = Σ_{k,h,w} |W_{c,k,h,w}|
        """
        weight = conv.weight.data
        importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
        return importance
    
    def _prune_layer(self, model: nn.Module, layer_name: str, prune_ratio: float):
        """
        对指定层进行结构化剪枝
        
        参数:
            model: 网络模型
            layer_name: 层名称
            prune_ratio: 剪枝比例 [0, 1)
        """
        # 获取目标层
        parts = layer_name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        
        if not isinstance(module, nn.Conv2d):
            return
        
        # 计算重要性
        importance = self._compute_l1_importance(module)
        n_channels = len(importance)
        n_prune = int(n_channels * prune_ratio)
        
        if n_prune == 0:
            return
        
        # 选择要剪枝的通道
        _, prune_indices = torch.topk(importance, n_prune, largest=False)
        keep_indices = [i for i in range(n_channels) if i not in prune_indices]
        keep_indices = torch.tensor(keep_indices, device=module.weight.device)
        
        # 修改权重（通过置零模拟剪枝）
        with torch.no_grad():
            mask = torch.zeros_like(module.weight)
            mask[keep_indices] = 1.0
            module.weight.data *= mask
            if module.bias is not None:
                bias_mask = torch.zeros_like(module.bias)
                bias_mask[keep_indices] = 1.0
                module.bias.data *= bias_mask
    
    @torch.no_grad()
    def _evaluate(self, model: nn.Module) -> float:
        """评估模型性能（返回DICE分数）"""
        model.eval()
        total_dice = 0.0
        n_samples = 0
        
        for batch in self.dataloader:
            if isinstance(batch, (list, tuple)):
                fixed, moving = batch[0].to(self.device), batch[1].to(self.device)
                if len(batch) > 2:
                    target = batch[2].to(self.device)
                else:
                    target = fixed
            else:
                continue
            
            # 前向传播
            try:
                outputs = model(fixed, moving)
                if isinstance(outputs, dict):
                    warped = outputs.get('warped_lvl0', outputs.get('warped', None))
                else:
                    warped = outputs
                
                if warped is not None:
                    # 计算DICE
                    dice = self._compute_dice(warped, target)
                    total_dice += dice * fixed.size(0)
                    n_samples += fixed.size(0)
            except Exception as e:
                continue
        
        return total_dice / max(n_samples, 1)
    
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor, 
                      threshold: float = 0.5) -> float:
        """计算DICE系数"""
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()
        
        intersection = (pred_bin * target_bin).sum()
        union = pred_bin.sum() + target_bin.sum()
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return dice.item()
    
    def analyze(self, pruning_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> Dict:
        """
        执行敏感度分析
        
        参数:
            pruning_ratios: 要测试的剪枝比例列表
        返回:
            sensitivity_results: 各层敏感度结果
        """
        print("\n" + "=" * 70)
        print("敏感度分析开始")
        print("=" * 70)
        
        # 获取基准性能
        baseline_dice = self._evaluate(self.model)
        print(f"基准DICE: {baseline_dice:.4f}")
        
        prunable_layers = self._get_prunable_layers()
        print(f"发现 {len(prunable_layers)} 个可剪枝层")
        
        results = {}
        
        for layer_name in prunable_layers:
            print(f"\n分析层: {layer_name}")
            dice_drop_rates = []
            
            for ratio in pruning_ratios:
                # 创建模型副本
                model_copy = copy.deepcopy(self.model)
                
                # 执行剪枝
                self._prune_layer(model_copy, layer_name, ratio)
                
                # 评估
                dice = self._evaluate(model_copy)
                dice_drop = baseline_dice - dice
                dice_drop_rates.append(dice_drop)
                
                print(f"  剪枝 {ratio*100:.0f}%: DICE降低 {dice_drop:.4f}")
                
                del model_copy
                torch.cuda.empty_cache()
            
            # 计算敏感度得分（斜率）
            if len(pruning_ratios) >= 2:
                sensitivity = np.polyfit(pruning_ratios, dice_drop_rates, 1)[0]
            else:
                sensitivity = dice_drop_rates[0] / pruning_ratios[0] if pruning_ratios[0] > 0 else 0
            
            results[layer_name] = {
                'sensitivity': sensitivity,
                'dice_drops': dict(zip(pruning_ratios, dice_drop_rates))
            }
        
        self.sensitivity_scores = results
        return results
    
    def get_pruning_config(self, sensitivity_threshold: float = 0.02) -> Dict[str, float]:
        """
        根据敏感度分析结果生成剪枝配置
        
        参数:
            sensitivity_threshold: 敏感度阈值，低于此值可以大幅剪枝
        返回:
            pruning_config: {layer_name: pruning_ratio}
        """
        config = {}
        
        for layer_name, result in self.sensitivity_scores.items():
            sensitivity = result['sensitivity']
            
            if sensitivity < 0.005:
                config[layer_name] = 0.4  # 非常不敏感，可剪枝40%
            elif sensitivity < 0.01:
                config[layer_name] = 0.3  # 较不敏感，剪枝30%
            elif sensitivity < sensitivity_threshold:
                config[layer_name] = 0.2  # 中等敏感，剪枝20%
            else:
                config[layer_name] = 0.0  # 敏感层，保留
        
        return config


class StructuredPruner:
    """
    结构化剪枝器
    
    采用L1-norm通道剪枝策略
    """
    
    def __init__(self, model: nn.Module, pruning_config: Dict[str, float]):
        """
        参数:
            model: 原始模型
            pruning_config: 剪枝配置 {layer_name: pruning_ratio}
        """
        self.model = model
        self.pruning_config = pruning_config
        
    def _get_layer_by_name(self, model: nn.Module, name: str) -> nn.Module:
        """根据名称获取层"""
        parts = name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        return module
    
    def _set_layer_by_name(self, model: nn.Module, name: str, new_module: nn.Module):
        """根据名称设置层"""
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], new_module)
    
    def prune(self) -> nn.Module:
        """
        执行结构化剪枝
        
        返回:
            pruned_model: 剪枝后的模型
        """
        print("\n" + "=" * 70)
        print("执行结构化剪枝")
        print("=" * 70)
        
        pruned_model = copy.deepcopy(self.model)
        
        # 记录通道映射，用于处理后续层
        channel_mappings = {}
        
        for layer_name, prune_ratio in self.pruning_config.items():
            if prune_ratio == 0:
                continue
                
            try:
                layer = self._get_layer_by_name(pruned_model, layer_name)
                
                if not isinstance(layer, nn.Conv2d):
                    continue
                
                # 计算L1重要性
                weight = layer.weight.data
                importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
                
                n_channels = len(importance)
                n_keep = int(n_channels * (1 - prune_ratio))
                n_keep = max(n_keep, 1)  # 至少保留1个通道
                
                # 选择保留的通道
                _, keep_indices = torch.topk(importance, n_keep)
                keep_indices = sorted(keep_indices.tolist())
                
                # 创建新的卷积层
                new_conv = nn.Conv2d(
                    in_channels=layer.in_channels,
                    out_channels=n_keep,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=layer.bias is not None
                )
                
                # 复制权重
                new_conv.weight.data = layer.weight.data[keep_indices]
                if layer.bias is not None:
                    new_conv.bias.data = layer.bias.data[keep_indices]
                
                # 替换层
                self._set_layer_by_name(pruned_model, layer_name, new_conv)
                
                channel_mappings[layer_name] = keep_indices
                print(f"剪枝 {layer_name}: {n_channels} → {n_keep} 通道 ({prune_ratio*100:.0f}%)")
                
            except Exception as e:
                print(f"剪枝层 {layer_name} 时出错: {e}")
        
        return pruned_model


# ===================== 3.3.3 误差感知知识蒸馏 =====================

class ErrorAwareDistillationLoss(nn.Module):
    """
    误差感知知识蒸馏损失
    
    对低置信区域的蒸馏给予更高权重，确保学生模型在这些区域
    能够复现教师模型的误差修正行为
    """
    
    def __init__(self, alpha: float = 0.5, beta_feature: float = 0.5, 
                 beta_output: float = 1.0, beta_ea: float = 0.3,
                 temperature: float = 4.0):
        """
        参数:
            alpha: 误差感知权重放大系数
            beta_feature: 特征蒸馏损失权重
            beta_output: 输出蒸馏损失权重  
            beta_ea: 误差感知蒸馏损失权重
            temperature: 蒸馏温度参数
        """
        super().__init__()
        self.alpha = alpha
        self.beta_feature = beta_feature
        self.beta_output = beta_output
        self.beta_ea = beta_ea
        self.temperature = temperature
        
    def compute_error_aware_weight(self, confidence_map: torch.Tensor) -> torch.Tensor:
        """
        计算误差感知权重
        
        W_EA(u,v) = 1 + α * (1 - C(u,v))
        
        参数:
            confidence_map: 置信度图 [B, 1, H, W]
        返回:
            weight_map: 权重图 [B, 1, H, W]
        """
        return 1.0 + self.alpha * (1.0 - confidence_map)
    
    def feature_distillation_loss(self, student_features: torch.Tensor, 
                                  teacher_features: torch.Tensor) -> torch.Tensor:
        """
        标准特征蒸馏损失
        
        L_feature = ||F_S - F_T||²
        """
        # 如果尺寸不匹配，进行插值
        if student_features.shape != teacher_features.shape:
            teacher_features = F.interpolate(
                teacher_features, 
                size=student_features.shape[2:],
                mode='bilinear', 
                align_corners=True
            )
        
        return F.mse_loss(student_features, teacher_features)
    
    def output_distillation_loss(self, student_output: torch.Tensor,
                                 teacher_output: torch.Tensor) -> torch.Tensor:
        """
        输出蒸馏损失
        
        L_output = ||φ_S - φ_T||²
        """
        if student_output.shape != teacher_output.shape:
            teacher_output = F.interpolate(
                teacher_output,
                size=student_output.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        return F.mse_loss(student_output, teacher_output)
    
    def error_aware_distillation_loss(self, student_features: torch.Tensor,
                                      teacher_features: torch.Tensor,
                                      confidence_map: torch.Tensor) -> torch.Tensor:
        """
        误差感知蒸馏损失
        
        L_EA = Σ W_EA(u,v) * ||F_S(u,v) - F_T(u,v)||²
        
        低置信区域的特征蒸馏更严格
        """
        # 对齐尺寸
        if student_features.shape != teacher_features.shape:
            teacher_features = F.interpolate(
                teacher_features,
                size=student_features.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        # 计算误差感知权重
        if confidence_map.shape[2:] != student_features.shape[2:]:
            confidence_map = F.interpolate(
                confidence_map,
                size=student_features.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        weight_map = self.compute_error_aware_weight(confidence_map)
        
        # 加权MSE损失
        diff_squared = (student_features - teacher_features) ** 2
        weighted_loss = (weight_map * diff_squared).mean()
        
        return weighted_loss
    
    def forward(self, student_outputs: Dict, teacher_outputs: Dict,
                confidence_map: Optional[torch.Tensor] = None,
                task_loss: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算总蒸馏损失
        
        L_total = L_task + β1*L_feature + β2*L_output + β3*L_EA
        
        参数:
            student_outputs: 学生模型输出字典
            teacher_outputs: 教师模型输出字典
            confidence_map: 置信度图
            task_loss: 任务损失（配准损失）
        返回:
            losses: 各项损失字典
        """
        losses = {}
        
        # 特征蒸馏损失
        if 'features' in student_outputs and 'features' in teacher_outputs:
            feature_loss = 0.0
            for s_feat, t_feat in zip(student_outputs['features'], 
                                      teacher_outputs['features']):
                feature_loss += self.feature_distillation_loss(s_feat, t_feat)
            losses['feature_loss'] = feature_loss * self.beta_feature
        
        # 输出蒸馏损失
        s_flow = student_outputs.get('flow_lvl0', student_outputs.get('flow'))
        t_flow = teacher_outputs.get('flow_lvl0', teacher_outputs.get('flow'))
        
        if s_flow is not None and t_flow is not None:
            losses['output_loss'] = self.output_distillation_loss(s_flow, t_flow) * self.beta_output
        
        # 误差感知蒸馏损失
        if confidence_map is not None and 'features' in student_outputs:
            ea_loss = 0.0
            for s_feat, t_feat in zip(student_outputs['features'],
                                      teacher_outputs['features']):
                ea_loss += self.error_aware_distillation_loss(s_feat, t_feat, confidence_map)
            losses['ea_loss'] = ea_loss * self.beta_ea
        
        # 总损失
        total_loss = sum(losses.values())
        if task_loss is not None:
            total_loss = total_loss + task_loss
            losses['task_loss'] = task_loss
        
        losses['total_loss'] = total_loss
        
        return losses


class KnowledgeDistillationTrainer:
    """
    知识蒸馏训练器
    
    实现两阶段训练策略:
    阶段1：热身 - 学生模型快速逼近教师
    阶段2：微调 - 强化误差感知能力
    """
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 train_loader, val_loader, device='cuda'):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.distill_loss = ErrorAwareDistillationLoss()
        
    def train(self, num_epochs: int = 100, warmup_epochs: int = 30,
              lr_warmup: float = 1e-4, lr_finetune: float = 1e-5):
        """
        执行两阶段蒸馏训练
        
        参数:
            num_epochs: 总训练轮数
            warmup_epochs: 热身阶段轮数
            lr_warmup: 热身阶段学习率
            lr_finetune: 微调阶段学习率
        """
        print("\n" + "=" * 70)
        print("知识蒸馏训练开始")
        print("=" * 70)
        
        # 阶段1：热身
        print(f"\n阶段1：热身训练 ({warmup_epochs}轮)")
        print("-" * 40)
        
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr_warmup)
        self.distill_loss.beta_feature = 0.8  # 高特征蒸馏权重
        self.distill_loss.beta_output = 0.8
        self.distill_loss.beta_ea = 0.1  # 低误差感知权重
        
        for epoch in range(warmup_epochs):
            train_loss = self._train_epoch(optimizer)
            val_dice = self._validate()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{warmup_epochs}: Loss={train_loss:.4f}, Val DICE={val_dice:.4f}")
        
        # 阶段2：微调
        print(f"\n阶段2：误差感知微调 ({num_epochs - warmup_epochs}轮)")
        print("-" * 40)
        
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr_finetune)
        self.distill_loss.beta_feature = 0.3  # 降低特征蒸馏权重
        self.distill_loss.beta_output = 0.5
        self.distill_loss.beta_ea = 0.5  # 提高误差感知权重
        
        best_dice = 0.0
        for epoch in range(warmup_epochs, num_epochs):
            train_loss = self._train_epoch(optimizer)
            val_dice = self._validate()
            
            if val_dice > best_dice:
                best_dice = val_dice
                # 保存最佳模型
                torch.save(self.student.state_dict(), 'best_student_model.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, "
                      f"Val DICE={val_dice:.4f}, Best={best_dice:.4f}")
        
        print(f"\n训练完成! 最佳验证DICE: {best_dice:.4f}")
        return best_dice
    
    def _train_epoch(self, optimizer) -> float:
        """训练一个epoch"""
        self.student.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                fixed = batch[0].to(self.device)
                moving = batch[1].to(self.device)
                confidence_map = batch[2].to(self.device) if len(batch) > 2 else None
            else:
                continue
            
            optimizer.zero_grad()
            
            # 教师前向传播（不计算梯度）
            with torch.no_grad():
                teacher_outputs = self.teacher(fixed, moving, confidence_map)
            
            # 学生前向传播
            student_outputs = self.student(fixed, moving, confidence_map)
            
            # 计算蒸馏损失
            losses = self.distill_loss(student_outputs, teacher_outputs, confidence_map)
            
            loss = losses['total_loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def _validate(self) -> float:
        """验证模型"""
        self.student.eval()
        total_dice = 0.0
        n_samples = 0
        
        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                fixed = batch[0].to(self.device)
                moving = batch[1].to(self.device)
            else:
                continue
            
            outputs = self.student(fixed, moving)
            
            if isinstance(outputs, dict):
                warped = outputs.get('warped_lvl0', outputs.get('warped'))
            else:
                warped = outputs
            
            if warped is not None:
                # 简化DICE计算
                dice = self._compute_dice(warped, fixed)
                total_dice += dice * fixed.size(0)
                n_samples += fixed.size(0)
        
        return total_dice / max(n_samples, 1)
    
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算DICE"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        return (2.0 * intersection / (pred_flat.sum() + target_flat.sum() + 1e-6)).item()


# ===================== 3.3.4 动态噪声抑制（DNS）模块 =====================

class NoiseDetector(nn.Module):
    """
    实时噪声水平检测器
    
    计算输入特征图的局部方差作为噪声指示
    """
    
    def __init__(self, kernel_size: int = 3, threshold_percentile: float = 95.0):
        """
        参数:
            kernel_size: 局部方差计算窗口大小
            threshold_percentile: 噪声阈值百分位数
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 用于计算局部均值的卷积核
        self.register_buffer(
            'mean_kernel',
            torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        )
        
        # 噪声阈值（训练过程中自适应更新）
        self.register_buffer('noise_threshold', torch.tensor(0.1))
        self.threshold_percentile = threshold_percentile
        
        # 用于EMA更新阈值的统计量
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.momentum = 0.1
        
    def compute_local_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算局部方差
        
        Var_3×3(F_ij) = E[F²] - E[F]²
        """
        B, C, H, W = x.shape
        
        # 扩展均值核以匹配输入通道
        mean_kernel = self.mean_kernel.expand(C, 1, -1, -1)
        
        # 计算局部均值 E[F]
        local_mean = F.conv2d(x, mean_kernel, padding=self.padding, groups=C)
        
        # 计算局部平方均值 E[F²]
        local_sq_mean = F.conv2d(x ** 2, mean_kernel, padding=self.padding, groups=C)
        
        # 方差 = E[F²] - E[F]²
        local_var = local_sq_mean - local_mean ** 2
        local_var = F.relu(local_var)  # 确保非负
        
        return local_var
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        检测噪声水平
        
        返回:
            noise_level: 归一化噪声水平 [0, 1]
            is_noisy: 是否超过阈值
        """
        local_var = self.compute_local_variance(x)
        
        # 全局噪声水平：平均局部方差
        noise_level = local_var.mean()
        
        # 训练时更新阈值
        if self.training:
            with torch.no_grad():
                # EMA更新
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                    self.momentum * noise_level
                self.running_var = (1 - self.momentum) * self.running_var + \
                                   self.momentum * ((noise_level - self.running_mean) ** 2)
                
                # 更新阈值（均值 + 2σ）
                self.noise_threshold = self.running_mean + 2 * torch.sqrt(self.running_var)
        
        is_noisy = noise_level > self.noise_threshold
        
        return noise_level, is_noisy


class MultiScaleGaussianFilter(nn.Module):
    """
    多尺度高斯滤波模块
    
    当检测到高噪声时，使用多尺度高斯滤波去噪
    """
    
    def __init__(self, channels: int, sigmas: List[float] = [0.5, 1.0, 2.0]):
        """
        参数:
            channels: 输入通道数
            sigmas: 各尺度的高斯核标准差
        """
        super().__init__()
        self.sigmas = sigmas
        self.n_scales = len(sigmas)
        
        # 为每个尺度创建高斯核
        self.gaussian_kernels = nn.ModuleList()
        for sigma in sigmas:
            kernel = self._create_gaussian_kernel(sigma, channels)
            self.gaussian_kernels.append(kernel)
        
        # 多尺度特征融合
        self.fusion = nn.Conv2d(channels * self.n_scales, channels, kernel_size=1)
        
    def _create_gaussian_kernel(self, sigma: float, channels: int) -> nn.Conv2d:
        """创建高斯滤波卷积层"""
        kernel_size = int(6 * sigma + 1) | 1  # 确保为奇数
        
        # 生成高斯核
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # 扩展到多通道
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1).clone()
        
        # 创建深度可分离卷积
        conv = nn.Conv2d(channels, channels, kernel_size, 
                        padding=kernel_size // 2, groups=channels, bias=False)
        conv.weight.data = kernel_2d
        conv.weight.requires_grad = False  # 固定高斯核
        
        return conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度高斯滤波
        
        F_filtered = F_in + Conv1×1([GB_σ1(F); GB_σ2(F); GB_σ3(F)])
        """
        filtered_list = []
        
        for gaussian_conv in self.gaussian_kernels:
            filtered = gaussian_conv(x)
            filtered_list.append(filtered)
        
        # 拼接多尺度特征
        multi_scale = torch.cat(filtered_list, dim=1)
        
        # 融合
        fused = self.fusion(multi_scale)
        
        # 残差连接
        output = x + fused
        
        return output


class DynamicNoiseSuppressionModule(nn.Module):
    """
    动态噪声抑制（DNS）模块
    
    实现对电磁干扰的自适应滤波:
    - 噪声检测器实时估计噪声水平
    - 低噪声时直接跳过滤波分支
    - 高噪声时激活多尺度滤波 + MRI引导去噪
    """
    
    def __init__(self, channels: int, memory_channels: int = None,
                 gamma_base: float = 0.1, gamma_max: float = 0.3):
        """
        参数:
            channels: 特征通道数
            memory_channels: MRI记忆特征通道数（用于MRI引导去噪）
            gamma_base: 基础MRI融合系数
            gamma_max: 最大MRI融合系数
        """
        super().__init__()
        
        self.noise_detector = NoiseDetector()
        self.gaussian_filter = MultiScaleGaussianFilter(channels)
        
        self.gamma_base = gamma_base
        self.gamma_max = gamma_max
        
        # MRI记忆投影（可选）
        if memory_channels is not None and memory_channels != channels:
            self.memory_proj = nn.Conv2d(memory_channels, channels, kernel_size=1)
        else:
            self.memory_proj = nn.Identity()
        
    def forward(self, x: torch.Tensor, 
                mri_memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        动态噪声抑制
        
        参数:
            x: 输入特征 [B, C, H, W]
            mri_memory: MRI记忆特征 [B, C', H, W]（可选）
        返回:
            output: 去噪后的特征
            info: 噪声检测信息
        """
        # 噪声检测
        noise_level, is_noisy = self.noise_detector(x)
        
        info = {
            'noise_level': noise_level.item() if isinstance(noise_level, torch.Tensor) else noise_level,
            'is_noisy': is_noisy.item() if isinstance(is_noisy, torch.Tensor) else is_noisy,
            'threshold': self.noise_detector.noise_threshold.item()
        }
        
        if not is_noisy:
            # 低噪声：直接跳过
            return x, info
        
        # 高噪声：多尺度高斯滤波
        filtered = self.gaussian_filter(x)
        
        # MRI引导去噪增强（如果提供了MRI记忆）
        if mri_memory is not None:
            # 投影MRI特征
            mri_feat = self.memory_proj(mri_memory)
            
            # 确保尺寸匹配
            if mri_feat.shape[2:] != filtered.shape[2:]:
                mri_feat = F.interpolate(mri_feat, size=filtered.shape[2:],
                                        mode='bilinear', align_corners=True)
            
            # 动态融合系数
            threshold = self.noise_detector.noise_threshold
            gamma = min(
                self.gamma_max,
                self.gamma_base * (noise_level - threshold) / (threshold + 1e-6)
            )
            gamma = max(0, gamma)
            
            # MRI引导融合
            output = filtered + gamma * mri_feat
            info['gamma'] = gamma.item() if isinstance(gamma, torch.Tensor) else gamma
        else:
            output = filtered
        
        return output, info


# ===================== 3.3.5 边缘设备部署优化 =====================

class EdgeDeploymentOptimizer:
    """
    边缘设备部署优化器
    
    提供TensorRT优化配置和部署工具
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 2, 256, 256)):
        self.model = model
        self.input_shape = input_shape
        
    def export_onnx(self, output_path: str = 'model.onnx', 
                    dynamic_batch: bool = False) -> str:
        """
        导出ONNX模型
        
        参数:
            output_path: 输出路径
            dynamic_batch: 是否支持动态batch
        返回:
            output_path: 导出的模型路径
        """
        print(f"导出ONNX模型到: {output_path}")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 创建dummy输入
        dummy_fixed = torch.randn(self.input_shape[0], 1, 
                                  self.input_shape[2], self.input_shape[3]).to(device)
        dummy_moving = torch.randn_like(dummy_fixed)
        
        # 动态轴配置
        if dynamic_batch:
            dynamic_axes = {
                'fixed': {0: 'batch_size'},
                'moving': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes = None
        
        try:
            torch.onnx.export(
                self.model,
                (dummy_fixed, dummy_moving),
                output_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['fixed', 'moving'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            print(f"ONNX模型导出成功!")
            return output_path
        except Exception as e:
            print(f"ONNX导出失败: {e}")
            return None
    
    def get_tensorrt_config(self) -> Dict:
        """
        获取TensorRT优化配置
        
        返回:
            trt_config: TensorRT配置字典
        """
        return {
            'precision': 'FP16',  # 半精度推理
            'workspace_size': 1 << 30,  # 1GB workspace
            'max_batch_size': 1,  # 单帧推理
            'optimization_profile': {
                'input': [
                    (1, 2, 256, 256),  # min shape
                    (1, 2, 256, 256),  # optimal shape
                    (1, 2, 256, 256)   # max shape
                ]
            },
            'layer_fusion': True,  # 层融合优化
            'kernel_auto_tuning': True  # 自动调优
        }
    
    def estimate_performance(self, device: str = 'jetson_orin') -> Dict:
        """
        估计在目标设备上的性能
        
        参数:
            device: 目标设备 ('jetson_orin', 'jetson_xavier_nx', 'jetson_nano')
        """
        # 设备性能参数
        device_specs = {
            'jetson_orin': {'tops': 200, 'memory_gb': 32, 'power_w': 60},
            'jetson_xavier_nx': {'tops': 10, 'memory_gb': 8, 'power_w': 20},
            'jetson_nano': {'tops': 0.5, 'memory_gb': 4, 'power_w': 10}
        }
        
        if device not in device_specs:
            print(f"未知设备: {device}")
            return {}
        
        specs = device_specs[device]
        
        # 估计延迟（基于经验公式）
        # 原网络在RTX 3090上约45.6ms
        rtx_latency = 45.6  # ms
        rtx_tops = 35.58
        
        # 简单的线性缩放估计
        estimated_latency = rtx_latency * (rtx_tops / specs['tops'])
        
        # FP16优化约提升1.5-2倍
        estimated_latency_fp16 = estimated_latency / 1.8
        
        return {
            'device': device,
            'specs': specs,
            'estimated_latency_fp32': f"{estimated_latency:.1f} ms",
            'estimated_latency_fp16': f"{estimated_latency_fp16:.1f} ms",
            'meets_realtime': estimated_latency_fp16 < 50  # <50ms为实时
        }


# ===================== 集成测试 =====================

class LightweightErrorAwareNet(nn.Module):
    """
    轻量化误差感知配准网络
    
    集成所有3.3节优化:
    - 结构化剪枝后的网络结构
    - 误差感知机制
    - 动态噪声抑制
    """
    
    def __init__(self, in_channels: int = 2, base_channels: int = 16,
                 pruning_ratio: float = 0.3):
        super().__init__()
        
        # 计算剪枝后的通道数
        c1 = int(base_channels * (1 - pruning_ratio))
        c2 = int(base_channels * 2 * (1 - pruning_ratio))
        c3 = int(base_channels * 4 * (1 - pruning_ratio * 0.5))  # 高层保留更多
        c4 = base_channels * 8  # 瓶颈层不剪枝
        
        # 轻量化编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, 3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        
        # DNS模块
        self.dns = DynamicNoiseSuppressionModule(c4)
        
        # 轻量化解码器
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c4, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c3, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c2, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        
        # 流场预测头
        self.flow_head = nn.Conv2d(c1, 2, 3, padding=1)
        
    def forward(self, fixed: torch.Tensor, moving: torch.Tensor,
                confidence_map: Optional[torch.Tensor] = None) -> Dict:
        """前向传播"""
        # 拼接输入
        x = torch.cat([fixed, moving], dim=1)
        
        # 编码
        features = self.encoder(x)
        
        # DNS去噪
        features_denoised, dns_info = self.dns(features)
        
        # 解码
        decoded = self.decoder(features_denoised)
        
        # 流场预测
        flow = self.flow_head(decoded)
        
        # 空间变换
        warped = self._warp(moving, flow)
        
        return {
            'warped': warped,
            'warped_lvl0': warped,
            'flow': flow,
            'flow_lvl0': flow,
            'dns_info': dns_info
        }
    
    def _warp(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """空间变换"""
        B, C, H, W = img.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0).expand(B, -1, -1, -1)
        
        # 添加流场
        flow_perm = flow.permute(0, 2, 3, 1)
        flow_norm = flow_perm * 2.0 / torch.tensor([W - 1, H - 1], device=img.device)
        new_grid = grid + flow_norm
        
        # 采样
        warped = F.grid_sample(img, new_grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)
        return warped


def test_lightweight_optimization():
    """测试轻量化优化模块"""
    print("\n" + "=" * 70)
    print("3.3 轻量化与抗噪优化模块测试")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 测试轻量化网络
    print("\n[测试1] 轻量化网络")
    model = LightweightErrorAwareNet(pruning_ratio=0.3).to(device)
    
    B, H, W = 2, 256, 256
    fixed = torch.randn(B, 1, H, W).to(device)
    moving = torch.randn(B, 1, H, W).to(device)
    
    outputs = model(fixed, moving)
    print(f"输入尺寸: {fixed.shape}")
    print(f"输出warped尺寸: {outputs['warped'].shape}")
    print(f"输出flow尺寸: {outputs['flow'].shape}")
    print(f"DNS信息: {outputs['dns_info']}")
    
    # 2. 测试模型分析
    print("\n[测试2] 模型计算量分析")
    profiler = ModelProfiler(model, input_size=(1, 2, H, W))
    profile_results = profiler.profile()
    print(f"总FLOPs: {profile_results['flops_giga']:.2f} G")
    print(f"总参数量: {profile_results['params_mega']:.2f} M")
    
    # 3. 测试DNS模块
    print("\n[测试3] 动态噪声抑制模块")
    dns = DynamicNoiseSuppressionModule(channels=128).to(device)
    
    # 正常特征
    normal_feat = torch.randn(1, 128, 32, 32).to(device)
    out_normal, info_normal = dns(normal_feat)
    print(f"正常输入 - 噪声级别: {info_normal['noise_level']:.4f}, 是否噪声: {info_normal['is_noisy']}")
    
    # 噪声特征
    noisy_feat = normal_feat + torch.randn_like(normal_feat) * 2
    out_noisy, info_noisy = dns(noisy_feat)
    print(f"噪声输入 - 噪声级别: {info_noisy['noise_level']:.4f}, 是否噪声: {info_noisy['is_noisy']}")
    
    # 4. 测试边缘部署优化
    print("\n[测试4] 边缘设备性能估计")
    optimizer = EdgeDeploymentOptimizer(model)
    
    for device_name in ['jetson_orin', 'jetson_xavier_nx', 'jetson_nano']:
        perf = optimizer.estimate_performance(device_name)
        print(f"{device_name}: 预计延迟(FP16) {perf['estimated_latency_fp16']}, "
              f"满足实时: {'✓' if perf['meets_realtime'] else '✗'}")
    
    print("\n测试完成!")
    return model


if __name__ == "__main__":
    test_lightweight_optimization()

