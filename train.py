import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler  # 添加混合精度训练支持

from model import MultiResolutionRegNet
from losses import DualSimilarityLoss
from data_loader import get_data_loaders
from utils import save_image, save_checkpoint, load_checkpoint, compute_dice_score, apply_colormap


def train(config):
    """
    训练多尺度配准网络

    参数:
        config: 训练配置字典
    """
    # 创建保存目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)

    # 设置设备并启用cudnn自动优化选择算法
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 根据输入动态选择最优算法
    print(f"使用设备: {device}")

    # 创建数据加载器
    train_loader, val_loader = get_data_loaders(
        root_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    print(f"训练数据批次数: {len(train_loader)}, 验证数据批次数: {len(val_loader)}")

    # 创建模型 - 修改为2通道输入(灰度CT和灰度MRI)
    model = MultiResolutionRegNet(in_channels=2).to(device)

    # 使用半精度加速训练
    scaler = GradScaler() if torch.cuda.is_available() else None

    # 如果有多GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)

    # 创建损失函数
    criterion = DualSimilarityLoss(
        alpha=config['alpha'],
        beta=config['beta']
    ).to(device)

    # 创建优化器并使用学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # 添加学习率调度器，每10个epoch衰减10%的学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 如果指定了预训练模型，加载checkpoint
    start_epoch = 0
    best_loss = float('inf')
    if config['resume']:
        start_epoch, best_loss = load_checkpoint(config['resume'], model, optimizer)
        print(f"从 epoch {start_epoch} 继续训练, 最佳损失: {best_loss:.4f}")

    # 创建TensorBoard记录器
    writer = SummaryWriter(log_dir=config['log_dir'])

    # 记录训练开始时间
    train_start_time = time.time()

    # 训练循环
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()

        # 训练一个epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler  # 添加混合精度训练scaler
        )

        # 仅在有限的epoch上进行完整验证，减少评估开销
        if epoch % config.get('val_frequency', 1) == 0:
            # 验证
            val_loss, val_metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                config=config
            )

            # 记录损失和指标
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/dice', val_metrics['dice'], epoch)
            writer.add_scalar('Metrics/ssim', val_metrics['ssim'], epoch)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }, is_best=True, save_path=config['checkpoint_dir'])
                print(f"保存最佳模型，损失: {best_loss:.4f}")

            # 每隔一定epoch保存一次checkpoint，而不是每个epoch都保存
            if (epoch + 1) % config.get('save_frequency_epochs', 5) == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }, is_best=False, save_path=config['checkpoint_dir'])

            # 计算并打印每个epoch的时间
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{config['num_epochs']} - "
                  f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                  f"DICE: {val_metrics['dice']:.4f}, SSIM: {val_metrics['ssim']:.4f}, "
                  f"耗时: {epoch_time:.1f}秒")
        else:
            # 仅训练不验证的epoch，只记录训练损失
            writer.add_scalar('Loss/train', train_loss, epoch)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{config['num_epochs']} - "
                  f"训练损失: {train_loss:.4f}, 耗时: {epoch_time:.1f}秒")

        # 更新学习率
        scheduler.step()

    # 计算总训练时间
    total_train_time = time.time() - train_start_time
    print(f"训练完成！总耗时: {total_train_time:.1f}秒")
    writer.close()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, scaler=None):
    """
    训练一个epoch

    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        config: 配置
        scaler: 混合精度训练的梯度缩放器

    返回:
        平均训练损失
    """
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1} Training")

    # 记录每个batch的处理时间
    batch_times = []

    for i, batch in progress_bar:
        batch_start = time.time()

        # 获取数据
        ct_imgs = batch['ct'].to(device, non_blocking=True)  # 使用non_blocking加速数据传输
        mri_imgs = batch['mri'].to(device, non_blocking=True)
        deformed_mri_imgs = batch['deformed_mri'].to(device, non_blocking=True)

        # 清零梯度
        optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True可以提高性能

        # 使用混合精度训练
        if scaler is not None:
            with autocast():
                # 前向传播
                outputs = model(ct_imgs, deformed_mri_imgs)

                # 计算损失
                loss_dict = criterion(ct_imgs, outputs['warped_lvl0'], outputs['flow_lvl0'])
                loss = loss_dict['total']

            # 使用scaler进行反向传播和优化器步进
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播
            outputs = model(ct_imgs, deformed_mri_imgs)

            # 计算损失
            loss_dict = criterion(ct_imgs, outputs['warped_lvl0'], outputs['flow_lvl0'])
            loss = loss_dict['total']

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

        # 更新累积损失
        epoch_loss += loss.item()

        # 计算当前batch耗时
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mi_loss': f"{loss_dict['mi'].item():.4f}",
            'mind_loss': f"{loss_dict['mind'].item():.4f}",
            'reg_loss': f"{loss_dict['reg'].item():.4f}",
            'batch_time': f"{batch_time:.3f}s"
        })

        # 每隔一定步数保存样本图像 - 减少保存频率以提高性能
        if i % config['save_frequency'] == 0 and epoch % config.get('save_img_frequency', 5) == 0:
            with torch.no_grad():
                # 保存样本图像，用于可视化
                # 确保使用同一个样本的所有图像，避免混淆
                for b in range(min(1, ct_imgs.size(0))):  # 每次只保存一个样本
                    # 获取文件名
                    filenames = batch['filenames']

                    # 从filenames数据结构中正确提取文件名
                    if isinstance(filenames, list) and len(filenames) == 3:
                        ct_name = filenames[0][b] if len(filenames[0]) > b else "unknown"
                        mri_name = filenames[1][b] if len(filenames[1]) > b else "unknown"
                        deformed_mri_name = filenames[2][b] if len(filenames[2]) > b else "unknown"

                        print(f"保存训练可视化图像 样本{b}: CT={ct_name}, MRI={mri_name}, 形变MRI={deformed_mri_name}")

                        # 保存灰度图像 - 原始数据
                        save_image(
                            imgs=[
                                ct_imgs[b].cpu(),
                                deformed_mri_imgs[b].cpu(),
                                outputs['warped_lvl0'][b].cpu(),
                                mri_imgs[b].cpu()
                            ],
                            titles=['CT', 'Deformed MRI', 'Registered MRI', 'Ground Truth MRI'],
                            save_path=os.path.join(config['result_dir'],
                                                   f"train_gray_ep{epoch + 1}_iter{i}_sample{b}.png"),
                            cmap='gray'
                        )

    # 计算平均损失和平均批次时间
    avg_loss = epoch_loss / len(train_loader)
    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"Epoch {epoch + 1} 平均批次时间: {avg_batch_time:.3f}秒")

    return avg_loss


def validate(model, val_loader, criterion, device, epoch, config):
    """
    验证模型

    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        config: 配置

    返回:
        平均验证损失和评估指标
    """
    model.eval()
    epoch_loss = 0
    all_dice_scores = []
    all_ssim_values = []

    # 减少验证时的日志输出以提高速度
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch + 1} Validation")

    with torch.no_grad():
        for i, batch in progress_bar:
            # 获取数据
            ct_imgs = batch['ct'].to(device, non_blocking=True)
            mri_imgs = batch['mri'].to(device, non_blocking=True)
            deformed_mri_imgs = batch['deformed_mri'].to(device, non_blocking=True)

            # 前向传播
            outputs = model(ct_imgs, deformed_mri_imgs)

            # 计算损失
            loss_dict = criterion(ct_imgs, outputs['warped_lvl0'], outputs['flow_lvl0'])
            loss = loss_dict['total']

            # 计算评估指标
            dice_score = compute_dice_score(outputs['warped_lvl0'], mri_imgs)
            ssim_value = compute_ssim(outputs['warped_lvl0'], mri_imgs)

            # 更新累积损失和指标
            epoch_loss += loss.item()
            all_dice_scores.append(dice_score)
            all_ssim_values.append(ssim_value)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{dice_score:.4f}",
                'ssim': f"{ssim_value:.4f}"
            })

            # 减少验证中的图像保存频率，仅保存最后一个epoch的图像
            if i % config['save_frequency'] == 0 and epoch == config['num_epochs'] - 1:
                # 保存样本图像，用于可视化
                for b in range(min(1, ct_imgs.size(0))):  # 只保存一个样本
                    # 获取文件名
                    filenames = batch['filenames']

                    # 从filenames数据结构中正确提取文件名
                    if isinstance(filenames, list) and len(filenames) == 3:
                        ct_name = filenames[0][b] if len(filenames[0]) > b else "unknown"
                        mri_name = filenames[1][b] if len(filenames[1]) > b else "unknown"
                        deformed_mri_name = filenames[2][b] if len(filenames[2]) > b else "unknown"

                        print(f"保存验证可视化图像 样本{b}: CT={ct_name}, MRI={mri_name}, 形变MRI={deformed_mri_name}")

                        # 保存灰度图像 - 原始数据
                        save_image(
                            imgs=[
                                ct_imgs[b].cpu(),
                                deformed_mri_imgs[b].cpu(),
                                outputs['warped_lvl0'][b].cpu(),
                                mri_imgs[b].cpu()
                            ],
                            titles=['CT', 'Deformed MRI', 'Registered MRI', 'Ground Truth MRI'],
                            save_path=os.path.join(config['result_dir'],
                                                   f"val_gray_ep{epoch + 1}_iter{i}_sample{b}.png"),
                            cmap='gray'
                        )

    # 计算平均损失和指标
    avg_loss = epoch_loss / len(val_loader)
    avg_dice = np.mean(all_dice_scores)
    avg_ssim = np.mean(all_ssim_values)

    metrics = {
        'dice': avg_dice,
        'ssim': avg_ssim
    }

    return avg_loss, metrics


def compute_ssim(x, y, window_size=11, size_average=True):
    """
    计算结构相似性指数（SSIM）- 修改为适应单通道灰度图像
    优化版本，减少计算量

    参数:
        x: 第一个图像 (B, 1, H, W)
        y: 第二个图像 (B, 1, H, W)
        window_size: 高斯窗口大小
        size_average: 是否在批次上平均

    返回:
        SSIM值
    """
    # 已经是灰度图，不需要转换
    x_gray = x
    y_gray = y

    # 下采样以加速计算
    if x.shape[2] > 64:  # 如果图像尺寸较大则进行下采样
        x_gray = F.avg_pool2d(x_gray, kernel_size=2, stride=2)
        y_gray = F.avg_pool2d(y_gray, kernel_size=2, stride=2)

    # 定义高斯窗口并缓存
    def _gaussian_window(window_size, sigma=1.5):
        gauss = torch.exp(
            -torch.arange(window_size).float().div(window_size // 2).pow(2).mul(2.0).div(2 * sigma * sigma)
        )
        return gauss / gauss.sum()

    # 创建1D高斯核
    gaussian_kernel = _gaussian_window(window_size)

    # 创建2D高斯核
    window = gaussian_kernel.unsqueeze(1) * gaussian_kernel.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).to(x.device)
    window = window.expand(1, 1, window_size, window_size).contiguous()

    # 填充图像
    pad = window_size // 2

    # 使用滑动窗口计算局部均值和方差
    mu1 = F.conv2d(
        F.pad(x_gray, [pad, pad, pad, pad], mode='replicate'),
        window,
        groups=1
    )
    mu2 = F.conv2d(
        F.pad(y_gray, [pad, pad, pad, pad], mode='replicate'),
        window,
        groups=1
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        F.pad(x_gray * x_gray, [pad, pad, pad, pad], mode='replicate'),
        window,
        groups=1
    ) - mu1_sq
    sigma2_sq = F.conv2d(
        F.pad(y_gray * y_gray, [pad, pad, pad, pad], mode='replicate'),
        window,
        groups=1
    ) - mu2_sq
    sigma12 = F.conv2d(
        F.pad(x_gray * y_gray, [pad, pad, pad, pad], mode='replicate'),
        window,
        groups=1
    ) - mu1_mu2

    # SSIM稳定性常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


if __name__ == "__main__":
    # 训练配置
    config = {
        'data_dir': './0426_data',  # 数据集根目录
        'checkpoint_dir': './0426_checkpoints',  # 检查点保存目录
        'log_dir': './logs',  # 日志保存目录
        'result_dir': './results',  # 结果保存目录
        'batch_size': 32,  # 批次大小增加，提高GPU利用率
        'num_workers': 0,  # 数据加载的工作线程数增加
        'learning_rate': 1e-4,  # 学习率略微提高
        'num_epochs': 11,  # 训练轮数
        'save_frequency': 20,  # 保存可视化结果的频率减少
        'save_frequency_epochs': 5,  # 每5个epoch保存一次模型
        'val_frequency': 5,  # 每2个epoch验证一次
        'save_img_frequency': 5,  # 每5个epoch保存一次图像
        'resume': None,  # 继续训练的检查点路径
        'alpha': 10.0,  # MIND损失权重
        'beta': 0.5  # 正则化损失权重
    }

    # 开始训练
    train(config)