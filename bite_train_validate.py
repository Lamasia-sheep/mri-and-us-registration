"""
BITE 数据集预训练与验证脚本
============================
使用 BITE 数据集 (真实临床 MRI-US 配对) 对配准模型进行:
  1. 预训练: 在真实 MR-US 数据上训练, 可用作后续 0426_data 微调的初始化
  2. 验证: 加载已有 checkpoint 在 BITE 上评估泛化能力
  3. 微调: 从 0426_data 的 checkpoint 出发, 在 BITE 上继续训练

配准方向: MR (fixed/目标) ← US (moving/待配准)
理由: MRI 解剖结构清晰、全脑覆盖，作为固定图像提供更好的监督信号和损失梯度。
"""

import os
import time
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from new_model import MultiResolutionRegNet  # 使用包含 SSM+LTM 的完整模型
from losses import DualSimilarityLoss
from bite_data_loader import get_bite_data_loaders
from utils import save_image, save_checkpoint, load_checkpoint, compute_dice_score


# ========================== 评估指标 ==========================

def compute_nmi(x, y, bins=64):
    """
    计算归一化互信息 (NMI)
    """
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()

    # 计算联合直方图
    hist_2d, _, _ = np.histogram2d(x_np, y_np, bins=bins)
    hist_2d = hist_2d / hist_2d.sum()

    # 边缘分布
    px = hist_2d.sum(axis=1)
    py = hist_2d.sum(axis=0)

    # 计算熵
    px_py = px[:, None] * py[None, :]
    nonzero = hist_2d > 0
    mi = (hist_2d[nonzero] * np.log(hist_2d[nonzero] / px_py[nonzero])).sum()

    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()

    nmi = 2 * mi / (hx + hy + 1e-10)
    return nmi


def compute_ssim(x, y, window_size=11):
    """计算 SSIM (与 train.py 中一致)"""
    x_gray = x
    y_gray = y

    if x.shape[2] > 64:
        x_gray = F.avg_pool2d(x_gray, kernel_size=2, stride=2)
        y_gray = F.avg_pool2d(y_gray, kernel_size=2, stride=2)

    def _gaussian_window(ws, sigma=1.5):
        gauss = torch.exp(
            -torch.arange(ws).float().div(ws // 2).pow(2).mul(2.0).div(2 * sigma * sigma)
        )
        return gauss / gauss.sum()

    gaussian_kernel = _gaussian_window(window_size)
    window = gaussian_kernel.unsqueeze(1) * gaussian_kernel.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).to(x.device)
    window = window.expand(1, 1, window_size, window_size).contiguous()

    pad = window_size // 2

    mu1 = F.conv2d(F.pad(x_gray, [pad]*4, mode='replicate'), window, groups=1)
    mu2 = F.conv2d(F.pad(y_gray, [pad]*4, mode='replicate'), window, groups=1)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(F.pad(x_gray * x_gray, [pad]*4, mode='replicate'), window, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(y_gray * y_gray, [pad]*4, mode='replicate'), window, groups=1) - mu2_sq
    sigma12 = F.conv2d(F.pad(x_gray * y_gray, [pad]*4, mode='replicate'), window, groups=1) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# ========================== 验证函数 ==========================

def validate_on_bite(model, val_loader, criterion, device, save_dir=None, max_vis=8):
    """
    在 BITE 验证集上评估模型。

    评估指标:
      - 配准前后的 NMI
      - 配准前后的 SSIM
      - 损失值
      - 变形场平滑度

    参数:
        model: 配准模型
        val_loader: BITE 验证数据加载器
        criterion: 损失函数
        device: 设备
        save_dir: 可视化结果保存目录
        max_vis: 最多保存多少组可视化
    """
    model.eval()
    all_losses = []
    all_nmi_before = []
    all_nmi_after = []
    all_ssim_before = []
    all_ssim_after = []
    vis_count = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("\n开始在 BITE 数据集上验证...")
    print("配准方向: MR (fixed) ← US (moving)")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="BITE 验证")):
            fixed_imgs = batch['fixed'].to(device)    # MRI 固定图像
            moving_imgs = batch['moving'].to(device)   # US 移动图像

            # 前向传播: model(fixed=MR, moving=US)
            outputs = model(fixed_imgs, moving_imgs)
            warped_us = outputs['warped_lvl0']
            flow = outputs['flow_lvl0']

            # 计算损失: 比较 fixed(MR) 和 warped_moving(warped US)
            loss_dict = criterion(fixed_imgs, warped_us, flow)
            all_losses.append(loss_dict['total'].item())

            # 计算配准前后的指标 (逐样本)
            for b in range(fixed_imgs.shape[0]):
                # 配准前: MR vs US
                nmi_before = compute_nmi(fixed_imgs[b], moving_imgs[b])
                ssim_before = compute_ssim(
                    fixed_imgs[b].unsqueeze(0), moving_imgs[b].unsqueeze(0)
                )

                # 配准后: MR vs Warped US
                nmi_after = compute_nmi(fixed_imgs[b], warped_us[b])
                ssim_after = compute_ssim(
                    fixed_imgs[b].unsqueeze(0), warped_us[b].unsqueeze(0)
                )

                all_nmi_before.append(nmi_before)
                all_nmi_after.append(nmi_after)
                all_ssim_before.append(ssim_before)
                all_ssim_after.append(ssim_after)

            # 可视化
            if save_dir and vis_count < max_vis:
                for b in range(min(1, fixed_imgs.shape[0])):
                    pid = batch['patient_id'][b]
                    sid = batch['slice_idx'][b].item()

                    save_image(
                        imgs=[
                            fixed_imgs[b].cpu(),
                            moving_imgs[b].cpu(),
                            warped_us[b].cpu(),
                            (fixed_imgs[b] - warped_us[b]).abs().cpu(),
                        ],
                        titles=[
                            f'MR (Fixed)\nP{pid} S{sid}',
                            'US (Moving)',
                            'Warped US',
                            '|MR - Warped US| After'
                        ],
                        save_path=os.path.join(save_dir, f"bite_val_P{pid}_S{sid}.png"),
                        cmap='gray'
                    )
                    vis_count += 1

    # 汇总结果
    results = {
        'loss': np.mean(all_losses),
        'nmi_before': np.mean(all_nmi_before),
        'nmi_after': np.mean(all_nmi_after),
        'nmi_improvement': np.mean(all_nmi_after) - np.mean(all_nmi_before),
        'ssim_before': np.mean(all_ssim_before),
        'ssim_after': np.mean(all_ssim_after),
        'ssim_improvement': np.mean(all_ssim_after) - np.mean(all_ssim_before),
        'num_samples': len(all_nmi_before)
    }

    print("\n" + "=" * 60)
    print("BITE 验证结果汇总")
    print("=" * 60)
    print(f"  验证样本数:     {results['num_samples']}")
    print(f"  平均损失:       {results['loss']:.4f}")
    print(f"  NMI  (配准前):  {results['nmi_before']:.4f}")
    print(f"  NMI  (配准后):  {results['nmi_after']:.4f}  "
          f"({'↑' if results['nmi_improvement'] > 0 else '↓'} {abs(results['nmi_improvement']):.4f})")
    print(f"  SSIM (配准前):  {results['ssim_before']:.4f}")
    print(f"  SSIM (配准后):  {results['ssim_after']:.4f}  "
          f"({'↑' if results['ssim_improvement'] > 0 else '↓'} {abs(results['ssim_improvement']):.4f})")
    print("=" * 60)

    return results


# ========================== 训练函数 ==========================

def train_on_bite(model, train_loader, val_loader, criterion, optimizer,
                  device, config, scaler=None):
    """
    在 BITE 数据集上训练 (预训练或微调)。

    参数:
        model: 配准模型
        train_loader: BITE 训练数据加载器
        val_loader: BITE 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        config: 训练配置
        scaler: 混合精度缩放器
    """
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    best_loss = float('inf')

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in progress:
            fixed_imgs = batch['fixed'].to(device, non_blocking=True)    # MR
            moving_imgs = batch['moving'].to(device, non_blocking=True)  # US

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model(fixed_imgs, moving_imgs)
                    loss_dict = criterion(fixed_imgs, outputs['warped_lvl0'], outputs['flow_lvl0'])
                    loss = loss_dict['total']
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(fixed_imgs, moving_imgs)
                loss_dict = criterion(fixed_imgs, outputs['warped_lvl0'], outputs['flow_lvl0'])
                loss = loss_dict['total']
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mi': f"{loss_dict['mi'].item():.4f}",
                'mind': f"{loss_dict['mind'].item():.4f}",
                'reg': f"{loss_dict['reg'].item():.4f}"
            })

        avg_loss = epoch_loss / batch_count
        scheduler.step()

        # 每几个 epoch 验证一次
        if (epoch + 1) % config.get('val_frequency', 5) == 0 or epoch == config['num_epochs'] - 1:
            val_results = validate_on_bite(
                model, val_loader, criterion, device,
                save_dir=os.path.join(config['result_dir'], f"bite_val_ep{epoch+1}")
            )
            val_loss = val_results['loss']

            print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"NMI={val_results['nmi_after']:.4f}, SSIM={val_results['ssim_after']:.4f}")

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'bite_results': val_results
                }, is_best=True, save_path=config['checkpoint_dir'])
                print(f"  保存最佳 BITE 模型, loss={best_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

    print(f"\nBITE 训练完成! 最佳验证损失: {best_loss:.4f}")


# ========================== 主入口 ==========================

def main():
    parser = argparse.ArgumentParser(description='BITE 数据集预训练与验证')
    parser.add_argument('--mode', type=str, default='validate',
                        choices=['pretrain', 'validate', 'finetune'],
                        help='运行模式: pretrain(从头预训练), validate(验证已有模型), finetune(微调)')
    parser.add_argument('--bite_dir', type=str, default='./group2 - png',
                        help='BITE 数据集目录')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径 (validate/finetune 时必须)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数 (pretrain/finetune)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--output_dir', type=str, default='./bite_output',
                        help='输出目录')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---- 加载 BITE 数据 ----
    train_loader, val_loader = get_bite_data_loaders(
        root_dir=args.bite_dir,
        batch_size=args.batch_size,
        num_workers=0
    )
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # ---- 创建模型 ----
    model = MultiResolutionRegNet(in_channels=2).to(device)

    # ---- 创建损失函数 ----
    criterion = DualSimilarityLoss(alpha=10.0, beta=0.5).to(device)

    # ---- 加载 checkpoint ----
    if args.checkpoint:
        print(f"加载 checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        print(f"  来自 epoch {ckpt.get('epoch', '?')}, loss={ckpt.get('best_loss', '?')}")

    # ---- 执行 ----
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'validate':
        # 仅验证
        if args.checkpoint is None:
            print("警告: validate 模式建议提供 --checkpoint，当前使用随机初始化模型")
        results = validate_on_bite(
            model, val_loader, criterion, device,
            save_dir=os.path.join(args.output_dir, 'bite_validate_results')
        )

    elif args.mode in ('pretrain', 'finetune'):
        # 训练
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 如果是 finetune 且有 checkpoint 的 optimizer state，加载它
        if args.mode == 'finetune' and args.checkpoint and 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                # 重设学习率
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr
                print(f"  已加载优化器状态, 学习率重设为 {args.lr}")
            except Exception as e:
                print(f"  加载优化器状态失败: {e}, 使用全新优化器")

        scaler = None
        if torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()

        config = {
            'num_epochs': args.epochs,
            'val_frequency': 5,
            'checkpoint_dir': os.path.join(args.output_dir, 'bite_checkpoints'),
            'result_dir': os.path.join(args.output_dir, 'bite_results'),
        }
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['result_dir'], exist_ok=True)

        train_on_bite(model, train_loader, val_loader, criterion, optimizer,
                      device, config, scaler)

    print("\n完成!")


if __name__ == "__main__":
    main()
