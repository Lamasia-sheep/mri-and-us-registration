"""
3.2.2 误差先验建模可视化模块

生成论文所需的图表:
- 图3.1 误差先验建模可视化结果 (6子图)
- 图3.2 误差先验模型验证实验结果 (2子图)
- 图3.3 区域对比图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from error_prior_modeling import (
    ErrorPriorModel, ErrorPriorModelValidator, analyze_spatial_nonuniformity,
    MagneticFieldErrorModel, ConfidenceMapGenerator, run_full_experiment
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_error_prior_modeling(model, confidence_map, sigma_2d, mf_model=None, save_path=None):
    """
    生成图3.1: 误差先验建模可视化结果
    
    包含6个子图:
    (a) 标定点空间分布与误差大小
    (b) 系统误差向量场（XY投影）
    (c) 误差随距离变化
    (d) 置信度图
    (e) 2D误差标准差图
    (f) 误差分量分布
    """
    fig = plt.figure(figsize=(16, 14))
    
    points = model.calibration_points
    systematic_errors = model.systematic_errors
    random_stds = model.random_stds
    
    # 计算总误差
    total_errors = np.sqrt(np.sum(systematic_errors ** 2, axis=1))
    
    # 获取磁场中心（如果有mf_model）
    if mf_model is None:
        # 尝试从workspace_size推断
        if hasattr(model, 'workspace_size'):
            center = np.array([model.workspace_size[0]/2, 
                             model.workspace_size[1]/2, 
                             model.workspace_size[2]/2])
        else:
            center = np.mean(points, axis=0)
    else:
        center = mf_model.center
    
    # 1. 标定点空间分布与误差大小
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=total_errors, cmap='RdYlGn_r', s=50, alpha=0.8)
    ax1.scatter(*center, color='red', s=200, marker='*', label='Magnetic Source')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('(a) Calibration Points Distribution\n& Total Error Magnitude')
    plt.colorbar(scatter, ax=ax1, label='Error (mm)', shrink=0.6)
    ax1.legend()
    
    # 2. 误差向量场（xy平面投影）
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.quiver(points[:, 0], points[:, 1],
              systematic_errors[:, 0], systematic_errors[:, 1],
              total_errors, cmap='RdYlGn_r', scale=50, alpha=0.8)
    ax2.scatter(center[0], center[1], color='red', s=200, marker='*', 
               label='Magnetic Source', zorder=5)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('(b) Systematic Error Vector Field\n(XY Projection)')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差随距离变化
    ax3 = fig.add_subplot(2, 3, 3)
    distances = np.linalg.norm(points - center, axis=1)
    ax3.scatter(distances, total_errors, c='steelblue', alpha=0.6, s=50)
    
    # 拟合趋势线
    z = np.polyfit(distances, total_errors, 2)
    p = np.poly1d(z)
    x_line = np.linspace(distances.min(), distances.max(), 100)
    ax3.plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend (quadratic)')
    
    ax3.set_xlabel('Distance to Magnetic Source (mm)')
    ax3.set_ylabel('Total Error (mm)')
    ax3.set_title('(c) Error vs. Distance from Source')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 置信度图
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(confidence_map, cmap='RdYlGn', aspect='auto',
                     extent=[0, 256*0.5, 256*0.5, 0])
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title('(d) Confidence Map C(u,v)')
    plt.colorbar(im4, ax=ax4, label='Confidence')
    
    # 5. 2D误差标准差图
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(sigma_2d, cmap='hot', aspect='auto',
                     extent=[0, 256*0.5, 256*0.5, 0])
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Y (mm)')
    ax5.set_title(r'(e) 2D Error Std $\sigma_{2D}$(u,v)')
    plt.colorbar(im5, ax=ax5, label='Std (mm)')
    
    # 6. 误差分量分布
    ax6 = fig.add_subplot(2, 3, 6)
    positions = np.arange(3)
    width = 0.35
    
    mean_systematic = np.mean(np.abs(systematic_errors), axis=0)
    mean_std = np.mean(random_stds, axis=0)
    
    bars1 = ax6.bar(positions - width/2, mean_systematic, width, 
                    label='Systematic Error', color='steelblue', alpha=0.8)
    bars2 = ax6.bar(positions + width/2, mean_std, width,
                    label='Random Std', color='coral', alpha=0.8)
    
    ax6.set_xlabel('Axis')
    ax6.set_ylabel('Error (mm)')
    ax6.set_title('(f) Error Components by Axis')
    ax6.set_xticks(positions)
    ax6.set_xticklabels(['X', 'Y', 'Z'])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax6.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax6.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  - 图表已保存到: {save_path}")
    
    return fig


def visualize_validation_results(validation_results, save_path=None):
    """
    生成图3.2: 误差先验模型验证实验结果
    
    包含2个子图:
    (a) 系统误差预测值vs真实值
    (b) 随机误差标准差预测值vs真实值
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 系统误差预测 vs 实际
    test_systematic = validation_results['systematic_error']['true_values']
    pred_systematic = validation_results['systematic_error']['pred_values']
    
    ax = axes[0]
    true_flat = test_systematic.flatten()
    pred_flat = pred_systematic.flatten()
    ax.scatter(true_flat, pred_flat, c='steelblue', alpha=0.6, s=50)
    lims = [min(true_flat.min(), pred_flat.min()) - 0.5,
            max(true_flat.max(), pred_flat.max()) + 0.5]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Systematic Error (mm)')
    ax.set_ylabel('Predicted Systematic Error (mm)')
    ax.set_title(f"(a) Systematic Error Prediction\nR={validation_results['systematic_error']['correlation']:.4f}, RMSE={validation_results['systematic_error']['rmse']:.4f}mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 随机误差标准差预测 vs 实际
    test_std = validation_results['random_error_std']['true_values']
    pred_std = validation_results['random_error_std']['pred_values']
    
    ax = axes[1]
    true_total = np.sqrt(np.sum(test_std ** 2, axis=1))
    pred_total = np.sqrt(np.sum(pred_std ** 2, axis=1))
    ax.scatter(true_total, pred_total, c='coral', alpha=0.6, s=50)
    lims = [min(true_total.min(), pred_total.min()) - 0.1,
            max(true_total.max(), pred_total.max()) + 0.1]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Total Std (mm)')
    ax.set_ylabel('Predicted Total Std (mm)')
    ax.set_title(f"(b) Random Error Std Prediction\nR={validation_results['random_error_std']['correlation']:.4f}, RMSE={validation_results['random_error_std']['rmse']:.4f}mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  - 验证实验图已保存到: {save_path}")
    
    return fig


def visualize_regional_comparison(model, center_threshold=120, mf_model=None, save_path=None):
    """
    可视化中心区域与边缘区域的误差对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    points = model.calibration_points
    systematic_errors = model.systematic_errors
    random_stds = model.random_stds
    
    # 获取磁场中心
    if mf_model is None:
        if hasattr(model, 'workspace_size'):
            center = np.array([model.workspace_size[0]/2, 
                             model.workspace_size[1]/2, 
                             model.workspace_size[2]/2])
        else:
            center = np.mean(points, axis=0)
    else:
        center = mf_model.center
    
    # 计算距离
    distances = np.linalg.norm(points - center, axis=1)
    center_mask = distances < center_threshold
    edge_mask = distances >= center_threshold
    
    # 总误差
    total_errors = np.sqrt(np.sum(systematic_errors ** 2, axis=1)) + \
                   np.sqrt(np.sum(random_stds ** 2, axis=1))
    
    # (a) 箱线图对比
    ax1 = axes[0]
    data = [total_errors[center_mask], total_errors[edge_mask]]
    bp = ax1.boxplot(data, labels=['中心区域\n(距磁源<120mm)', '边缘区域\n(距磁源≥120mm)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax1.set_ylabel('总误差 (mm)')
    ax1.set_title('(a) 区域误差对比箱线图')
    ax1.grid(True, alpha=0.3)
    
    # 添加均值标记
    means = [np.mean(d) for d in data]
    ax1.scatter([1, 2], means, color='red', marker='D', s=100, zorder=3, 
                label=f'均值: {means[0]:.2f}, {means[1]:.2f} mm')
    ax1.legend()
    
    # (b) 误差随距离变化
    ax2 = axes[1]
    ax2.scatter(distances, total_errors, c=total_errors, cmap='RdYlGn_r', 
                s=80, alpha=0.7, edgecolors='black')
    
    # 添加趋势线
    z = np.polyfit(distances, total_errors, 2)
    p = np.poly1d(z)
    x_line = np.linspace(distances.min(), distances.max(), 100)
    ax2.plot(x_line, p(x_line), 'b-', linewidth=2, label='趋势线(二次拟合)')
    
    # 标记中心区域边界
    ax2.axvline(x=center_threshold, color='red', linestyle='--', 
                linewidth=2, label=f'中心/边缘边界 ({center_threshold}mm)')
    
    ax2.set_xlabel('距磁源距离 (mm)')
    ax2.set_ylabel('总误差 (mm)')
    ax2.set_title('(b) 误差随距磁源距离变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  - 区域对比图已保存到: {save_path}")
    
    return fig


def generate_all_figures(output_dir='./figures'):
    """
    生成所有论文图表
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行实验
    print("=" * 60)
    print("开始生成3.2.2节所有图表")
    print("=" * 60)
    
    # 1. 创建磁场误差模型
    print("\n[步骤1] 初始化磁场误差模型...")
    mf_model = MagneticFieldErrorModel(workspace_size=(300, 300, 200))
    
    # 2. 生成标定点
    print("[步骤2] 生成50个标定点...")
    calibration_points = mf_model.generate_calibration_points(n_points=50, seed=42)
    
    # 3. 模拟测量
    print("[步骤3] 模拟测量过程...")
    measured_means, measured_stds, true_sys_errors = mf_model.simulate_measurements(
        calibration_points, n_measurements=10, seed=42
    )
    
    # 计算测量得到的系统误差
    measured_systematic_errors = measured_means - calibration_points
    
    # 4. 构建误差先验模型
    print("[步骤4] 构建RBF插值误差先验模型...")
    model = ErrorPriorModel(
        calibration_points=calibration_points,
        systematic_errors=measured_systematic_errors,
        random_stds=measured_stds
    )
    
    # 5. 生成置信度图
    print("[步骤5] 生成置信度图...")
    confidence_generator = ConfidenceMapGenerator(
        error_model=model,
        image_size=(256, 256),
        pixel_spacing=(0.5, 0.5)
    )
    
    plane_origin = np.array([100.0, 100.0, 100.0])
    plane_normal = np.array([0.0, 0.0, 1.0])
    plane_u = np.array([1.0, 0.0, 0.0])
    plane_v = np.array([0.0, 1.0, 0.0])
    
    confidence_map, sigma_2d, tau = confidence_generator.compute_confidence_map(
        plane_origin, plane_normal, plane_u, plane_v
    )
    
    # 6. 验证模型
    print("[步骤6] 模型验证...")
    validator = ErrorPriorModelValidator(model)
    validation_results = validator.validate_leave_out(train_ratio=0.8, seed=42)
    
    # 7. 分析空间非均匀性
    print("[步骤7] 分析空间非均匀性...")
    spatial_results = analyze_spatial_nonuniformity(model, center_threshold=120)
    
    # 生成图表
    print("\n生成图3.1...")
    visualize_error_prior_modeling(model, confidence_map, sigma_2d, mf_model=mf_model,
                                   save_path=f'{output_dir}/fig3_1_error_prior_modeling.png')
    plt.close()
    
    print("\n生成图3.2...")
    visualize_validation_results(validation_results,
                                save_path=f'{output_dir}/fig3_2_validation_results.png')
    plt.close()
    
    print("\n生成区域对比图...")
    visualize_regional_comparison(model, center_threshold=120, mf_model=mf_model,
                                 save_path=f'{output_dir}/fig3_3_regional_comparison.png')
    plt.close()
    
    print("\n所有图表生成完成!")
    return model, validation_results


if __name__ == "__main__":
    generate_all_figures()
