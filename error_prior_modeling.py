"""
3.2.2 误差先验建模
基于磁场辅助的多维度MRI与超声脑成像配准技术 - 第三章

本模块实现:
1. 误差数据采集协议
2. 误差分布的参数化建模 (RBF插值)
3. 置信度图的构建
4. 误差先验模型验证
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')


class MagneticFieldErrorModel:
    """
    磁场定位误差模型
    
    基于Aurora电磁跟踪系统的特性，模拟误差分布：
    - 空间非均匀性：中心区域误差小，边缘区域误差大
    - 方向相关性：沿磁场发射器轴向的误差小于垂直方向
    - 传感器噪声：0.5mm（Aurora系统标称精度）
    """
    
    def __init__(self, workspace_size=(300, 300, 200), center=None):
        """
        初始化工作空间
        
        参数：
            workspace_size: 工作空间大小 (mm)，对应磁场覆盖范围
            center: 磁场发射器中心位置 (mm)，如果为None则自动设置为工作空间中心
        """
        self.workspace_size = np.array(workspace_size)
        if center is None:
            self.center = np.array([workspace_size[0]/2, workspace_size[1]/2, workspace_size[2]/2])
        else:
            self.center = np.array(center)
        self.sensor_noise = 0.5  # Aurora系统标称精度 (mm)
        
    def generate_calibration_points(self, n_points=50, seed=42):
        """
        在磁场覆盖空间内均匀设置N个标定点
        
        参数:
            n_points: 标定点数量
            seed: 随机种子
        返回:
            calibration_points: (N, 3) 标定点坐标
        """
        np.random.seed(seed)
        
        # 使用分层采样确保空间均匀覆盖
        n_per_dim = int(np.ceil(n_points ** (1/3)))
        
        x = np.linspace(20, self.workspace_size[0] - 20, n_per_dim)
        y = np.linspace(20, self.workspace_size[1] - 20, n_per_dim)
        z = np.linspace(20, self.workspace_size[2] - 20, max(3, n_per_dim // 2))
        
        # 生成网格点
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # 随机选择n_points个点
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
            points = points[indices]
        
        # 添加少量随机扰动使分布更自然
        points += np.random.uniform(-5, 5, points.shape)
        points = np.clip(points, 10, self.workspace_size - 10)
        
        return points
    
    def compute_systematic_error(self, points):
        """
        计算系统误差 μ(x,y,z)
        
        系统误差特性：
        - 距离磁场中心越远，误差越大
        - 沿z轴（磁场主轴）方向误差较小
        """
        distances = np.linalg.norm(points - self.center, axis=1)
        max_dist = np.linalg.norm(self.workspace_size / 2)
        
        # 基础系统误差：与距离成正比
        base_error = 0.5 + 2.0 * (distances / max_dist) ** 1.5
        
        # 方向相关性：z轴方向误差较小
        z_deviation = np.abs(points[:, 2] - self.center[2]) / (self.workspace_size[2] / 2)
        direction_factor = 0.8 + 0.4 * z_deviation
        
        # 系统误差向量 (dx, dy, dz)
        # 误差方向大致指向远离中心
        error_direction = points - self.center
        error_direction = error_direction / (np.linalg.norm(error_direction, axis=1, keepdims=True) + 1e-6)
        
        systematic_error = error_direction * (base_error * direction_factor)[:, np.newaxis]
        
        # 添加一些各向异性：xy平面误差大于z轴
        systematic_error[:, 0:2] *= 1.2
        systematic_error[:, 2] *= 0.8
        
        return systematic_error
    
    def compute_random_error_std(self, points):
        """
        计算随机误差标准差 σ(x,y,z)
        
        随机误差特性：
        - 传感器基础噪声约0.5mm
        - 边缘区域噪声更大
        """
        distances = np.linalg.norm(points - self.center, axis=1)
        max_dist = np.linalg.norm(self.workspace_size / 2)
        
        # 基础随机误差
        base_std = self.sensor_noise
        
        # 空间变化的随机误差
        spatial_factor = 1.0 + 1.5 * (distances / max_dist) ** 2
        
        # 各向异性：xy平面噪声略大
        sigma = np.zeros((len(points), 3))
        sigma[:, 0] = base_std * spatial_factor * 1.1
        sigma[:, 1] = base_std * spatial_factor * 1.1
        sigma[:, 2] = base_std * spatial_factor * 0.9
        
        return sigma
    
    def simulate_measurements(self, points, n_measurements=10, seed=42):
        """
        模拟每个标定点的多次测量
        
        参数：
            points: 真值坐标
            n_measurements: 每个点的测量次数
            seed: 随机种子
        
        返回：
            measurements_mean: 测量均值 (N, 3)
            measurements_std: 测量标准差 (N, 3)
            systematic_error: 系统误差 (N, 3)
        """
        np.random.seed(seed)
        systematic_error = self.compute_systematic_error(points)
        random_std = self.compute_random_error_std(points)
        
        all_measurements = []
        for i in range(len(points)):
            # 对每个点进行n_measurements次测量
            noise = np.random.randn(n_measurements, 3) * random_std[i]
            point_measurements = points[i] + systematic_error[i] + noise
            all_measurements.append(point_measurements)
        
        all_measurements = np.array(all_measurements)
        
        # 计算测量均值和标准差
        measurements_mean = np.mean(all_measurements, axis=1)
        measurements_std = np.std(all_measurements, axis=1)
        
        return measurements_mean, measurements_std, systematic_error


class ErrorPriorModel:
    """
    基于RBF插值的误差先验模型
    
    对采集的误差数据进行统计建模，假设误差服从空间变化的高斯分布：
    ε(x,y,z) ~ N(μ(x,y,z), Σ(x,y,z))
    """
    
    def __init__(self, calibration_points=None, systematic_errors=None, random_stds=None, 
                 length_scale=None, workspace_size=(300, 300, 200), length_scale_factor=1.5):
        """
        初始化误差先验模型
        
        支持两种初始化方式：
        1. 直接提供标定点数据（新方式）
        2. 使用workspace_size和length_scale_factor（向后兼容）
        
        参数：
            calibration_points: 标定点真值坐标 (N, 3)
            systematic_errors: 系统误差 (N, 3)
            random_stds: 随机误差标准差 (N, 3)
            length_scale: RBF长度尺度参数
            workspace_size: 工作空间大小 (向后兼容)
            length_scale_factor: RBF长度尺度参数因子 (向后兼容)
        """
        # 向后兼容：如果没有提供数据，使用旧接口
        if calibration_points is None:
            self.workspace_size = workspace_size
            self.length_scale_factor = length_scale_factor
            self.calibration_points = None
            self.ground_truth = None
            self.systematic_errors = None
            self.random_errors_std = None
            self.length_scale = None
            self.mu_interpolators = None
            self.sigma_interpolators = None
            self.actual_systematic_errors = None
            return
        
        # 新接口：直接使用提供的数据
        self.calibration_points = np.array(calibration_points)
        self.systematic_errors = np.array(systematic_errors)
        self.random_stds = np.array(random_stds)
        
        # 计算标定点平均间距
        if length_scale is None:
            distances = cdist(self.calibration_points, self.calibration_points)
            np.fill_diagonal(distances, np.inf)
            avg_distance = np.mean(np.min(distances, axis=1))
            self.length_scale = avg_distance * 1.5
        else:
            self.length_scale = length_scale
        
        # 构建RBF插值器
        self._build_interpolators()
    
    def _build_interpolators(self):
        """
        使用RBF插值构建连续空间的误差模型
        """
        if self.calibration_points is None:
            return
            
        # 系统误差插值器（对每个分量分别插值）
        self.mu_interpolators = []
        for i in range(3):
            interp = RBFInterpolator(
                self.calibration_points,
                self.systematic_errors[:, i],
                kernel='gaussian',
                epsilon=1.0 / self.length_scale
            )
            self.mu_interpolators.append(interp)
        
        # 随机误差标准差插值器
        self.sigma_interpolators = []
        for i in range(3):
            interp = RBFInterpolator(
                self.calibration_points,
                self.random_stds[:, i],
                kernel='gaussian',
                epsilon=1.0 / self.length_scale
            )
            self.sigma_interpolators.append(interp)
    
    def generate_calibration_points(self, n_points=50, seed=42):
        """
        向后兼容：生成标定点（使用MagneticFieldErrorModel）
        """
        if not hasattr(self, 'workspace_size'):
            raise ValueError("请使用MagneticFieldErrorModel生成标定点，或直接提供calibration_points")
        
        mf_model = MagneticFieldErrorModel(workspace_size=self.workspace_size)
        self.calibration_points = mf_model.generate_calibration_points(n_points, seed)
        return self.calibration_points
    
    def simulate_measurements(self, n_repeats=10, sensor_noise_std=0.5, 
                             transform_error_std=1.0, seed=42):
        """
        向后兼容：模拟测量过程
        """
        if self.calibration_points is None:
            raise ValueError("请先调用generate_calibration_points生成标定点")
        
        mf_model = MagneticFieldErrorModel(workspace_size=self.workspace_size)
        measured_means, measured_stds, true_sys_errors = mf_model.simulate_measurements(
            self.calibration_points, n_repeats, seed
        )
        
        self.ground_truth = self.calibration_points.copy()
        self.actual_systematic_errors = measured_means - self.ground_truth
        self.random_errors_std = measured_stds
        
        # 计算RBF长度尺度参数
        pairwise_distances = cdist(self.calibration_points, self.calibration_points)
        np.fill_diagonal(pairwise_distances, np.inf)
        avg_spacing = np.mean(np.min(pairwise_distances, axis=1))
        self.length_scale = avg_spacing * self.length_scale_factor
        
        # 更新插值器
        self.systematic_errors = self.actual_systematic_errors
        self.random_stds = self.random_errors_std
        self._build_interpolators()
        
        return self.actual_systematic_errors, self.random_errors_std
    
    def fit_rbf_model(self):
        """
        向后兼容：拟合RBF模型（现在在__init__中自动完成）
        """
        if self.mu_interpolators is None:
            if self.calibration_points is None:
                raise ValueError("请先调用simulate_measurements生成测量数据")
            self._build_interpolators()
        print("RBF插值模型拟合完成")
    
    def predict_systematic_error(self, points):
        """
        预测任意位置的系统误差
        
        μ(x,y,z) = Σ w_i · φ(||P - P_i||)
        """
        if self.mu_interpolators is None:
            raise ValueError("请先拟合RBF模型")
        
        points = np.atleast_2d(points)
        mu = np.zeros((len(points), 3))
        for i in range(3):
            mu[:, i] = self.mu_interpolators[i](points)
        return mu
    
    def predict_random_std(self, points):
        """
        预测任意位置的随机误差标准差
        
        σ(x,y,z) = Σ v_i · φ(||P - P_i||)
        """
        if self.sigma_interpolators is None:
            raise ValueError("请先拟合RBF模型")
        
        points = np.atleast_2d(points)
        sigma = np.zeros((len(points), 3))
        for i in range(3):
            sigma[:, i] = self.sigma_interpolators[i](points)
        return np.clip(sigma, 0.1, None)  # 确保标准差为正
    
    def predict_total_error_std(self, points):
        """
        预测总误差标准差（用于置信度计算）
        """
        sigma = self.predict_random_std(points)
        return np.sqrt(np.sum(sigma ** 2, axis=1))
    
    def predict_error(self, query_points):
        """
        向后兼容：预测误差
        """
        systematic_error = self.predict_systematic_error(query_points)
        random_error_std = self.predict_random_std(query_points)
        return systematic_error, random_error_std
    
    def compute_confidence_map(self, image_size=(256, 256), 
                               plane_origin=None, plane_normal=None,
                               plane_u=None, plane_v=None,
                               pixel_spacing=(1.0, 1.0)):
        """
        向后兼容：计算置信度图
        """
        generator = ConfidenceMapGenerator(self, image_size, pixel_spacing)
        
        # 默认平面参数
        if plane_origin is None:
            if hasattr(self, 'workspace_size'):
                ws = self.workspace_size
            else:
                ws = (300, 300, 200)
            plane_origin = np.array([ws[0]/2 - image_size[1]*pixel_spacing[0]/2,
                                    ws[1]/2 - image_size[0]*pixel_spacing[1]/2,
                                    ws[2]/2])
        if plane_u is None:
            plane_u = np.array([1, 0, 0])
        if plane_v is None:
            plane_v = np.array([0, 1, 0])
        if plane_normal is None:
            plane_normal = np.array([0, 0, 1])
        
        return generator.compute_confidence_map(plane_origin, plane_normal, plane_u, plane_v)


class ConfidenceMapGenerator:
    """
    将三维误差分布转换为二维超声图像平面上的置信度图
    """
    
    def __init__(self, error_model, image_size=(256, 256), pixel_spacing=(0.5, 0.5)):
        """
        参数：
            error_model: 误差先验模型
            image_size: 超声图像大小 (H, W)
            pixel_spacing: 像素间距 (mm/pixel)
        """
        self.error_model = error_model
        self.image_size = image_size
        self.pixel_spacing = pixel_spacing
    
    def compute_confidence_map(self, plane_origin, plane_normal, plane_u, plane_v, tau=None):
        """
        计算超声图像平面上的置信度图
        
        参数：
            plane_origin: 平面原点在全局坐标系中的位置 (3,)
            plane_normal: 平面法向量 (3,)
            plane_u: 平面内u方向单位向量 (3,)
            plane_v: 平面内v方向单位向量 (3,)
            tau: 温度参数，控制置信度对误差的敏感程度
        
        返回：
            confidence_map: 置信度图 (H, W)
            sigma_2d: 二维误差标准差图 (H, W)
            tau: 温度参数
        """
        H, W = self.image_size
        du, dv = self.pixel_spacing
        
        # 生成超声图像平面上的像素坐标
        u_coords = np.arange(W) * du
        v_coords = np.arange(H) * dv
        uu, vv = np.meshgrid(u_coords, v_coords)
        
        # 将像素坐标转换为三维全局坐标
        # P_3D = plane_origin + u * plane_u + v * plane_v
        points_3d = (plane_origin + 
                     uu[:, :, np.newaxis] * plane_u + 
                     vv[:, :, np.newaxis] * plane_v)
        
        # 重塑为 (H*W, 3)
        points_flat = points_3d.reshape(-1, 3)
        
        # 预测每个像素位置的误差标准差
        sigma_total = self.error_model.predict_total_error_std(points_flat)
        sigma_2d = sigma_total.reshape(H, W)
        
        # 计算温度参数（中位数策略）
        if tau is None:
            tau = np.median(sigma_2d)
        
        # 计算归一化置信度
        # C(u,v) = exp(-σ²(u,v) / (2τ²))
        confidence_map = np.exp(-sigma_2d ** 2 / (2 * tau ** 2))
        
        return confidence_map, sigma_2d, tau


class ErrorPriorModelValidator:
    """
    误差先验模型验证类
    
    采用留出法验证RBF插值模型的预测准确性
    """
    
    def __init__(self, error_model):
        self.error_model = error_model
        
    def validate_leave_out(self, train_ratio=0.8, seed=42):
        """
        留出法验证
        
        参数:
            train_ratio: 训练集比例
            seed: 随机种子
        返回:
            results: 验证结果字典
        """
        if self.error_model.calibration_points is None:
            raise ValueError("误差模型未初始化，请先提供标定点数据")
        
        # 分割数据
        n_points = len(self.error_model.calibration_points)
        indices = np.arange(n_points)
        
        train_idx, val_idx = train_test_split(
            indices, train_size=train_ratio, random_state=seed
        )
        
        print(f"训练集: {len(train_idx)}个点, 验证集: {len(val_idx)}个点")
        
        # 使用训练集重新拟合模型
        train_points = self.error_model.calibration_points[train_idx]
        train_sys_errors = self.error_model.systematic_errors[train_idx]
        train_rand_std = self.error_model.random_stds[train_idx]
        
        # 创建新的模型用于验证
        val_model = ErrorPriorModel(
            calibration_points=train_points,
            systematic_errors=train_sys_errors,
            random_stds=train_rand_std
        )
        
        # 在验证集上预测
        val_points = self.error_model.calibration_points[val_idx]
        val_sys_errors_true = self.error_model.systematic_errors[val_idx]
        val_rand_std_true = self.error_model.random_stds[val_idx]
        
        val_sys_errors_pred = val_model.predict_systematic_error(val_points)
        val_rand_std_pred = val_model.predict_random_std(val_points)
        
        # 计算评估指标
        # 系统误差预测 - 计算总误差幅度
        sys_error_true_mag = np.linalg.norm(val_sys_errors_true, axis=1)
        sys_error_pred_mag = np.linalg.norm(val_sys_errors_pred, axis=1)
        
        # 相关系数
        sys_corr = np.corrcoef(sys_error_true_mag, sys_error_pred_mag)[0, 1]
        
        # RMSE
        sys_rmse = np.sqrt(mean_squared_error(sys_error_true_mag, sys_error_pred_mag))
        
        # 随机误差标准差预测
        rand_std_true_mag = np.linalg.norm(val_rand_std_true, axis=1)
        rand_std_pred_mag = np.linalg.norm(val_rand_std_pred, axis=1)
        
        rand_corr = np.corrcoef(rand_std_true_mag, rand_std_pred_mag)[0, 1]
        rand_rmse = np.sqrt(mean_squared_error(rand_std_true_mag, rand_std_pred_mag))
        
        results = {
            'systematic_error': {
                'correlation': sys_corr,
                'rmse': sys_rmse,
                'true_values': val_sys_errors_true,
                'pred_values': val_sys_errors_pred
            },
            'random_error_std': {
                'correlation': rand_corr,
                'rmse': rand_rmse,
                'true_values': val_rand_std_true,
                'pred_values': val_rand_std_pred
            },
            'train_indices': train_idx,
            'val_indices': val_idx
        }
        
        print("\n===== 误差先验模型验证结果 =====")
        print(f"系统误差预测 - 相关系数R: {sys_corr:.4f}, RMSE: {sys_rmse:.4f} mm")
        print(f"随机误差标准差预测 - 相关系数R: {rand_corr:.4f}, RMSE: {rand_rmse:.4f} mm")
        
        # 判断是否满足预期目标
        if sys_corr > 0.85:
            print(f"✓ 系统误差预测相关系数 {sys_corr:.4f} > 0.85，满足预期目标")
        else:
            print(f"✗ 系统误差预测相关系数 {sys_corr:.4f} < 0.85，未满足预期目标")
            
        if sys_rmse < 0.5:
            print(f"✓ 系统误差预测RMSE {sys_rmse:.4f} mm < 0.5 mm，满足预期目标")
        else:
            print(f"✗ 系统误差预测RMSE {sys_rmse:.4f} mm > 0.5 mm，未满足预期目标")
        
        return results


def analyze_spatial_nonuniformity(error_model, center_threshold=120):
    """
    分析误差的空间非均匀性
    
    参数:
        error_model: 误差先验模型
        center_threshold: 中心区域距磁源的距离阈值 (mm)
    返回:
        analysis_results: 分析结果字典
    """
    if error_model.calibration_points is None:
        raise ValueError("误差模型未初始化")
    
    # 计算每个标定点到磁源(假设在原点)的距离
    distances = np.linalg.norm(error_model.calibration_points, axis=1)
    
    # 分区
    center_mask = distances < center_threshold
    edge_mask = distances >= center_threshold
    
    # 计算各区域的平均误差
    total_errors = np.linalg.norm(error_model.systematic_errors, axis=1) + \
                   np.linalg.norm(error_model.random_stds, axis=1)
    
    center_error = np.mean(total_errors[center_mask]) if np.any(center_mask) else 0
    edge_error = np.mean(total_errors[edge_mask]) if np.any(edge_mask) else 0
    
    # 统计结果
    sys_error_mean = np.mean(np.abs(error_model.systematic_errors), axis=0)
    rand_error_mean = np.mean(error_model.random_stds, axis=0)
    
    total_sys_error = np.linalg.norm(sys_error_mean)
    total_rand_error = np.linalg.norm(rand_error_mean)
    
    results = {
        'center_region': {
            'threshold': center_threshold,
            'n_points': np.sum(center_mask),
            'mean_error': center_error,
            'characteristic': '高置信度'
        },
        'edge_region': {
            'n_points': np.sum(edge_mask),
            'mean_error': edge_error,
            'characteristic': '低置信度'
        },
        'overall_statistics': {
            'systematic_error_mean_x': sys_error_mean[0],
            'systematic_error_mean_y': sys_error_mean[1],
            'systematic_error_mean_z': sys_error_mean[2],
            'systematic_error_total': total_sys_error,
            'random_error_mean_x': rand_error_mean[0],
            'random_error_mean_y': rand_error_mean[1],
            'random_error_mean_z': rand_error_mean[2],
            'random_error_total': total_rand_error
        }
    }
    
    print("\n===== 误差空间分布分析 =====")
    print(f"中心区域 (距磁源<{center_threshold}mm): {results['center_region']['n_points']}个点, "
          f"平均误差: {center_error:.3f} mm")
    print(f"边缘区域 (距磁源≥{center_threshold}mm): {results['edge_region']['n_points']}个点, "
          f"平均误差: {edge_error:.3f} mm")
    print(f"\n系统误差均值: X={sys_error_mean[0]:.3f}, Y={sys_error_mean[1]:.3f}, "
          f"Z={sys_error_mean[2]:.3f}, 总计={total_sys_error:.3f} mm")
    print(f"随机误差均值: X={rand_error_mean[0]:.3f}, Y={rand_error_mean[1]:.3f}, "
          f"Z={rand_error_mean[2]:.3f}, 总计={total_rand_error:.3f} mm")
    
    return results


def run_full_experiment():
    """
    运行完整的3.2.2误差先验建模实验（向后兼容）
    """
    print("=" * 60)
    print("3.2.2 误差先验建模实验")
    print("=" * 60)
    
    # 1. 创建磁场误差模型
    print("\n[步骤1] 初始化磁场误差模型...")
    mf_model = MagneticFieldErrorModel(workspace_size=(300, 300, 200))
    
    # 2. 生成标定点
    print("\n[步骤2] 生成标定点...")
    calibration_points = mf_model.generate_calibration_points(n_points=50, seed=42)
    print(f"生成了 {len(calibration_points)} 个标定点")
    
    # 3. 模拟测量
    print("\n[步骤3] 模拟测量过程...")
    measured_means, measured_stds, true_sys_errors = mf_model.simulate_measurements(
        calibration_points, n_measurements=10, seed=42
    )
    
    # 计算测量得到的系统误差
    measured_systematic_errors = measured_means - calibration_points
    
    # 4. 构建误差先验模型
    print("\n[步骤4] 构建RBF插值误差先验模型...")
    model = ErrorPriorModel(
        calibration_points=calibration_points,
        systematic_errors=measured_systematic_errors,
        random_stds=measured_stds
    )
    print(f"RBF长度尺度参数: {model.length_scale:.2f} mm")
    
    # 5. 计算置信度图
    print("\n[步骤5] 计算置信度图...")
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
    print(f"温度参数τ: {tau:.3f} mm")
    print(f"置信度图范围: [{confidence_map.min():.3f}, {confidence_map.max():.3f}]")
    
    # 6. 分析空间非均匀性
    print("\n[步骤6] 分析空间非均匀性...")
    spatial_results = analyze_spatial_nonuniformity(model, center_threshold=120)
    
    # 7. 模型验证
    print("\n[步骤7] 模型验证 (留出法)...")
    validator = ErrorPriorModelValidator(model)
    validation_results = validator.validate_leave_out(train_ratio=0.8, seed=42)
    
    # 8. 输出误差数据表
    print("\n[步骤8] 生成误差数据表...")
    print_error_data_table(model, calibration_points, measured_systematic_errors)
    
    return model, confidence_map, validation_results, spatial_results


def print_error_data_table(model, calibration_points=None, systematic_errors=None, n_display=10):
    """
    打印误差数据表 (前n个标定点)
    """
    if calibration_points is None:
        calibration_points = model.calibration_points
    if systematic_errors is None:
        systematic_errors = model.systematic_errors
    
    print(f"\n表3.1 标定点误差数据（前{n_display}个，单位：mm）")
    print("-" * 90)
    print(f"{'ID':^4} {'真值X':^10} {'真值Y':^10} {'真值Z':^10} "
          f"{'误差dx':^10} {'误差dy':^10} {'误差dz':^10}")
    print("-" * 90)
    
    for i in range(min(n_display, len(calibration_points))):
        gt = calibration_points[i]
        err = systematic_errors[i]
        print(f"{i+1:^4} {gt[0]:^10.2f} {gt[1]:^10.2f} {gt[2]:^10.2f} "
              f"{err[0]:^10.3f} {err[1]:^10.3f} {err[2]:^10.3f}")
    print("-" * 90)


if __name__ == "__main__":
    model, confidence_map, val_results, spatial_results = run_full_experiment()
