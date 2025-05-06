import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import hausdorff_distance
from scipy.ndimage import label
from torch.utils.data import DataLoader
import torch.nn.functional as F

from new_model import MultiResolutionRegNet
from data_loader import MedicalImageDataset
from utils import load_checkpoint, save_image, compute_dice_score, apply_colormap, visualize_flow, gray_to_3channel


def compute_ssim(x, y, window_size=11, size_average=True):
    """
    计算结构相似性指数（SSIM）- 修改为适应单通道灰度图像

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

    # 定义高斯窗口
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


def compute_hd95(pred, target, threshold=0.5):
    """
    计算95%豪斯多夫距离 (HD95) - 修改为适应单通道灰度图像

    参数:
        pred: 预测图像 (B, 1, H, W)
        target: 目标图像 (B, 1, H, W)
        threshold: 二值化阈值

    返回:
        HD95值 (mm)
    """
    # 将图像转换为numpy数组
    pred_np = pred.detach().cpu().numpy()[0, 0]
    target_np = target.detach().cpu().numpy()[0, 0]

    # 二值化
    pred_binary = (pred_np > threshold).astype(np.uint8)
    target_binary = (target_np > threshold).astype(np.uint8)

    # 如果二值图像中没有前景像素，返回一个大的值
    if np.sum(pred_binary) == 0 or np.sum(target_binary) == 0:
        return 100.0

    # 计算豪斯多夫距离
    try:
        # 使用95%百分位的豪斯多夫距离
        from scipy.spatial.distance import directed_hausdorff

        # 获取轮廓点
        pred_points = np.argwhere(pred_binary)
        target_points = np.argwhere(target_binary)

        # 计算双向豪斯多夫距离
        if len(pred_points) > 0 and len(target_points) > 0:
            d1, _, _ = directed_hausdorff(pred_points, target_points)
            d2, _, _ = directed_hausdorff(target_points, pred_points)

            # 取两个方向的最大值
            hd_value = max(d1, d2)

            # 使用95%分位数代替最大值（HD95）
            if len(pred_points) > 20 and len(target_points) > 20:
                dists_1 = np.array([np.min(np.sqrt(np.sum((p - target_points) ** 2, axis=1))) for p in pred_points])
                dists_2 = np.array([np.min(np.sqrt(np.sum((p - pred_points) ** 2, axis=1))) for p in target_points])

                hd95 = max(np.percentile(dists_1, 95), np.percentile(dists_2, 95))
            else:
                hd95 = hd_value
        else:
            hd95 = 100.0  # 设置一个大值

        return hd95
    except Exception as e:
        print(f"计算HD95时出错: {e}")
        return 100.0


def compute_target_registration_error(pred, target):
    """
    计算目标配准误差 (TRE) - 修改为适应单通道灰度图像，提高特征点匹配的准确性

    参数:
        pred: 预测图像 (B, 1, H, W)
        target: 目标图像 (B, 1, H, W)

    返回:
        目标配准误差 (mm)和使用的特征点数量
    """
    # 将图像转换为numpy数组
    pred_np = pred.detach().cpu().numpy()[0, 0]  # 单通道灰度图像
    target_np = target.detach().cpu().numpy()[0, 0]  # 单通道灰度图像

    # 转换为8位无符号整数图像（CV2需要）
    pred_np_uint8 = np.uint8(pred_np * 255)
    target_np_uint8 = np.uint8(target_np * 255)

    try:
        # 使用ORB特征检测器（比Harris更稳定）
        orb = cv2.ORB_create(nfeatures=100)

        # 检测关键点和描述符
        kp1, des1 = orb.detectAndCompute(pred_np_uint8, None)
        kp2, des2 = orb.detectAndCompute(target_np_uint8, None)

        print(f"ORB特征检测 - 预测图像: {len(kp1)}个点, 目标图像: {len(kp2)}个点")

        # 如果检测到的特征点太少，使用边缘检测
        if len(kp1) < 10 or len(kp2) < 10 or des1 is None or des2 is None:
            # 使用Canny边缘检测
            pred_edges = cv2.Canny(pred_np_uint8, 100, 200)
            target_edges = cv2.Canny(target_np_uint8, 100, 200)

            # 找到非零像素位置作为特征点
            pred_points = np.argwhere(pred_edges)
            target_points = np.argwhere(target_edges)

            print(f"Canny边缘检测 - 预测图像: {len(pred_points)}个点, 目标图像: {len(target_points)}个点")

            # 如果有足够的边缘点
            if len(pred_points) > 10 and len(target_points) > 10:
                # 转换为浮点数进行计算
                pred_points = pred_points.astype(np.float32)
                target_points = target_points.astype(np.float32)

                # 创建点对应关系（为每个预测点找到目标中最近的点）
                matched_points = []
                # 限制点数量避免计算过慢
                max_points = min(100, len(pred_points))
                pred_indices = np.random.choice(len(pred_points), max_points, replace=False)

                for idx in pred_indices:
                    p = pred_points[idx]
                    # 找到目标中最近的点
                    distances = np.sqrt(np.sum((target_points - p) ** 2, axis=1))
                    nearest_idx = np.argmin(distances)
                    matched_points.append((p, target_points[nearest_idx]))

                # 计算匹配点之间的欧氏距离
                distances = [np.sqrt(np.sum((p[0] - p[1]) ** 2)) for p in matched_points]
                tre = np.mean(distances)
                num_points = len(matched_points)

                print(f"边缘点匹配 - 使用了{num_points}对匹配点")
            else:
                tre = 20.0
                num_points = 0
        else:
            # 使用BFMatcher匹配特征描述符
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # 按距离排序匹配结果
            matches = sorted(matches, key=lambda x: x.distance)

            # 限制匹配点数量
            max_matches = min(50, len(matches))
            good_matches = matches[:max_matches]

            print(f"ORB特征匹配 - 匹配点数量: {len(good_matches)}个")

            if len(good_matches) > 0:
                # 获取匹配点坐标
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                # 计算匹配点之间的欧氏距离
                distances = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1))

                # 计算平均距离（TRE）
                tre = np.mean(distances)
                num_points = len(good_matches)
            else:
                tre = 20.0
                num_points = 0

        return tre, num_points

    except Exception as e:
        print(f"计算TRE时出错: {e}")
        return 20.0, 0  # 返回一个较大的错误值和0个特征点


def compute_jacobian_fold_percentage(flow):
    """
    计算雅可比行列式小于等于0的比例

    参数:
        flow: 变形场 (B, 2, H, W)

    返回:
        雅可比行列式小于等于0的像素百分比
    """
    # 转为CPU
    flow = flow.detach().cpu()

    # 获取尺寸
    B, C, H, W = flow.size()

    # 计算雅可比矩阵的行列式
    # 初始化梯度张量
    dx_u = torch.zeros_like(flow[:, 0, :, :])
    dy_u = torch.zeros_like(flow[:, 0, :, :])
    dx_v = torch.zeros_like(flow[:, 1, :, :])
    dy_v = torch.zeros_like(flow[:, 1, :, :])

    # 计算x方向梯度 (正向差分)
    dx_u[:, :, :-1] = flow[:, 0, :, 1:] - flow[:, 0, :, :-1]
    dx_v[:, :, :-1] = flow[:, 1, :, 1:] - flow[:, 1, :, :-1]

    # 计算y方向梯度 (正向差分)
    dy_u[:, :-1, :] = flow[:, 0, 1:, :] - flow[:, 0, :-1, :]
    dy_v[:, :-1, :] = flow[:, 1, 1:, :] - flow[:, 1, :-1, :]

    # 计算雅可比行列式: J = (1+du/dx)*(1+dv/dy) - (du/dy)*(dv/dx)
    jacobian_det = (1 + dx_u) * (1 + dy_v) - dy_u * dx_v

    # 计算雅可比行列式小于等于0的像素数量
    neg_jacobian = (jacobian_det <= 0).float()

    # 计算像素百分比 (去除边缘像素)
    valid_pixels = (H - 1) * (W - 1) * B
    percentage = 100.0 * torch.sum(neg_jacobian) / valid_pixels

    return percentage.item()


def test(config):
    """
    测试模型性能

    参数:
        config: 测试配置字典
    """
    # 创建保存目录
    os.makedirs(config['result_dir'], exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型 - 修改为2通道输入(灰度CT和灰度MRI)
    model = MultiResolutionRegNet(in_channels=2).to(device)

    # 加载训练好的模型
    print(f"从 {config['checkpoint_path']} 加载模型...")
    _, best_loss = load_checkpoint(config['checkpoint_path'], model)
    print(f"模型加载完成，最佳损失: {best_loss}")
    model.eval()

    # 数据变换 - 修改为单通道灰度图像的归一化
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 创建测试数据集
    test_dataset = MedicalImageDataset(
        root_dir=config['data_dir'],
        is_train=False,
        transform=transform
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 测试时使用批次大小为1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )

    # 用于存储评估指标的列表
    dice_scores = []
    hd95_values = []
    target_registration_errors = []
    ssim_values = []
    jacobian_folds = []

    # 存储文件名信息
    all_ct_files = []
    all_mri_files = []
    all_deformed_files = []

    print(f"开始测试，共 {len(test_loader)} 个样本")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            # 获取数据
            ct_imgs = batch['ct'].to(device)
            mri_imgs = batch['mri'].to(device)
            deformed_mri_imgs = batch['deformed_mri'].to(device)
            filenames = batch['filenames']

            # 正确提取文件名
            if isinstance(filenames, list) and len(filenames) == 3:
                # 预期格式: [['file1.png'], ['file2.png'], ['file3.png']]
                ct_name = filenames[0][0] if len(filenames[0]) > 0 else "unknown"
                mri_name = filenames[1][0] if len(filenames[1]) > 0 else "unknown"
                deformed_mri_name = filenames[2][0] if len(filenames[2]) > 0 else "unknown"
            elif isinstance(filenames, tuple) and len(filenames) == 3:
                # 预期格式: (['file1.png'], ['file2.png'], ['file3.png'])
                ct_name = filenames[0][0] if len(filenames[0]) > 0 else "unknown"
                mri_name = filenames[1][0] if len(filenames[1]) > 0 else "unknown"
                deformed_mri_name = filenames[2][0] if len(filenames[2]) > 0 else "unknown"
            else:
                print(f"警告: 未能解析文件名格式: {filenames}")
                ct_name = "unknown"
                mri_name = "unknown"
                deformed_mri_name = "unknown"

            # 存储文件名信息
            all_ct_files.append(ct_name)
            all_mri_files.append(mri_name)
            all_deformed_files.append(deformed_mri_name)

            # 打印文件名
            print(f"处理测试样本 {i}: CT={ct_name}, MRI={mri_name}, 形变MRI={deformed_mri_name}")

            # 前向传播
            outputs = model(ct_imgs, deformed_mri_imgs)

            # 获取配准结果和变形场
            registered_mri = outputs['warped_lvl0']
            flow_field = outputs['flow_lvl0']

            # 计算指标
            # 1. Dice系数
            dice = compute_dice_score(registered_mri, mri_imgs)
            dice_scores.append(dice)

            # 2. 95%豪斯多夫距离 (Hd95)
            hd95 = compute_hd95(registered_mri, mri_imgs)
            hd95_values.append(hd95)

            # 3. 目标配准误差
            tre, num_points = compute_target_registration_error(registered_mri, mri_imgs)
            target_registration_errors.append(tre)

            # 4. 结构相似性指数
            ssim = compute_ssim(registered_mri, mri_imgs)
            ssim_values.append(ssim)

            # 5. 雅可比行列式小于等于0的比例
            fold_percentage = compute_jacobian_fold_percentage(flow_field)
            jacobian_folds.append(fold_percentage)

            # 获取变形场可视化（彩色）
            flow_viz = visualize_flow(flow_field[0].cpu())

            # 转换单通道图像为三通道灰度图像
            ct_3ch = gray_to_3channel(ct_imgs[0].cpu())
            deformed_mri_3ch = gray_to_3channel(deformed_mri_imgs[0].cpu())
            registered_mri_3ch = gray_to_3channel(registered_mri[0].cpu())
            mri_3ch = gray_to_3channel(mri_imgs[0].cpu())

            # 保存三通道灰度可视化
            color_save_path = os.path.join(config['result_dir'], f"test_sample_{i}_3ch_gray.png")
            save_image(
                imgs=[
                    ct_3ch,
                    deformed_mri_3ch,
                    registered_mri_3ch,
                    mri_3ch,
                    flow_viz  # 流场保持彩色
                ],
                titles=['CT', 'Deformed MRI', 'Registered MRI', 'Ground Truth MRI', 'Deformation Field'],
                save_path=color_save_path,
                use_3ch_gray=False  # 已经转换为3通道，不需要再次转换
            )

            # 保存单通道灰度图像
            gray_save_path = os.path.join(config['result_dir'], f"test_sample_{i}_gray.png")
            save_image(
                imgs=[
                    ct_imgs[0].cpu(),
                    deformed_mri_imgs[0].cpu(),
                    registered_mri[0].cpu(),
                    mri_imgs[0].cpu(),
                    flow_viz  # 流场保持彩色
                ],
                titles=['CT', 'Deformed MRI', 'Registered MRI', 'Ground Truth MRI', 'Deformation Field'],
                save_path=gray_save_path,
                cmap='gray'
            )

            # 额外保存一个只包含MRI相关图像的对比图（三通道灰度）
            mri_compare_path = os.path.join(config['result_dir'], f"test_sample_{i}_mri_compare.png")
            save_image(
                imgs=[
                    deformed_mri_3ch,
                    registered_mri_3ch,
                    mri_3ch
                ],
                titles=['Deformed MRI', 'Registered MRI', 'Ground Truth MRI'],
                save_path=mri_compare_path
            )

            # 打印当前样本的指标和文件名
            print(
                f"样本 {i} (CT={ct_name}, MRI={mri_name}, 形变MRI={deformed_mri_name}): DICE={dice:.4f}, HD95={hd95:.4f} mm, TRE={tre:.4f} mm, SSIM={ssim:.4f}, Fold%={fold_percentage:.4f}%")

    # 计算并打印平均指标
    avg_dice = np.mean(dice_scores)
    avg_hd95 = np.mean(hd95_values)
    avg_tre = np.mean(target_registration_errors)
    avg_ssim = np.mean(ssim_values)
    avg_fold = np.mean(jacobian_folds)

    std_dice = np.std(dice_scores)
    std_hd95 = np.std(hd95_values)
    std_tre = np.std(target_registration_errors)
    std_ssim = np.std(ssim_values)
    std_fold = np.std(jacobian_folds)

    print("\n===== 测试结果 =====")
    print(f"平均DICE系数: {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"平均HD95: {avg_hd95:.4f} ± {std_hd95:.4f} mm")
    print(f"平均目标配准误差: {avg_tre:.4f} ± {std_tre:.4f} mm")
    print(f"平均SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

    # 保存指标为CSV文件
    import csv
    csv_path = os.path.join(config['result_dir'], 'test_metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sample', 'CT', 'MRI', 'Deformed_MRI', 'DICE', 'HD95 (mm)', 'TRE (mm)', 'SSIM', 'Fold%'])
        for i in range(len(dice_scores)):
            writer.writerow([
                i,
                all_ct_files[i],
                all_mri_files[i],
                all_deformed_files[i],
                dice_scores[i],
                hd95_values[i],
                target_registration_errors[i],
                ssim_values[i],
                jacobian_folds[i]
            ])
        writer.writerow(['Average', '', '', '', avg_dice, avg_hd95, avg_tre, avg_ssim, avg_fold])
        writer.writerow(['Std', '', '', '', std_dice, std_hd95, std_tre, std_ssim, std_fold])

    print(f"指标已保存到 {csv_path}")


if __name__ == "__main__":
    # 测试配置
    config = {
        'data_dir': './0426_data',  # 数据集根目录
        'result_dir': './new_test_results',  # 结果保存目录
        'checkpoint_path': './0426_new_checkpoints/best_model.pth',  # 训练好的模型路径
        'num_workers': 4  # 数据加载的工作线程数
    }

    # 开始测试
    test(config)

    '''
    原始
    == == = 测试结果 == == =
    平均DICE系数: 0.7223 ± 0.0999
平均HD95: 1.9859 ± 0.2611 mm
平均目标配准误差: 3.0522 ± 1.1839 mm
平均SSIM: 0.7735 ± 0.0958
指标已保存到 ./test_results\test_metrics.csv
    '''

    '''
    new
    ===== 测试结果 =====
平均DICE系数: 0.7261 ± 0.1159
平均HD95: 1.2950 ± 0.3508 mm
平均目标配准误差: 2.4959 ± 1.1434 mm
平均SSIM: 0.7685 ± 0.1114
指标已保存到 ./new_test_results\test_metrics.csv
    '''