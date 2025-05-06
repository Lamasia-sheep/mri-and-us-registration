import os
import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def elastic_transform(image, alpha, sigma, random_state=None):
    """
    对图像进行弹性形变

    参数:
    - image: 输入图像
    - alpha: 形变强度
    - sigma: 高斯滤波的标准差，控制形变平滑度
    - random_state: 随机种子，用于重现结果

    返回:
    - 形变后的图像
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # 生成随机位移场
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # 生成网格
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # 将位移场应用到网格上
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # 对图像进行形变
    distorted_image = map_coordinates(image, indices, order=1).reshape(shape)

    return distorted_image


def affine_transform(image, max_rotation=10, max_translation=20, max_scale=0.2):
    """
    对图像进行仿射变换

    参数:
    - image: 输入图像
    - max_rotation: 最大旋转角度(度)
    - max_translation: 最大平移距离(像素)
    - max_scale: 最大缩放因子

    返回:
    - 变换后的图像
    """
    rows, cols = image.shape

    # 随机旋转角度
    angle = random.uniform(-max_rotation, max_rotation)

    # 随机平移
    tx = random.uniform(-max_translation, max_translation)
    ty = random.uniform(-max_translation, max_translation)

    # 随机缩放
    scale = random.uniform(1 - max_scale, 1 + max_scale)

    # 计算旋转矩阵
    center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 添加平移
    M[0, 2] += tx
    M[1, 2] += ty

    # 应用仿射变换
    transformed_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    return transformed_image


def generate_deformed_images(input_folder, output_folder, num_images_per_original=5,
                             elastic_alpha_range=(800, 1200), elastic_sigma_range=(8, 12),
                             affine_rotation_range=(5, 15), affine_translation_range=(10, 30),
                             affine_scale_range=(0.1, 0.3)):
    """
    生成形变的MRI图像并保存

    参数:
    - input_folder: 输入MRI图像的文件夹路径
    - output_folder: 输出形变图像的文件夹路径
    - num_images_per_original: 每张原始图像要生成的形变图像数量
    - elastic_alpha_range: 弹性形变强度范围
    - elastic_sigma_range: 弹性形变平滑度范围
    - affine_rotation_range: 仿射变换最大旋转角度范围(度)
    - affine_translation_range: 仿射变换最大平移距离范围(像素)
    - affine_scale_range: 仿射变换最大缩放因子范围
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有MRI图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"找到{len(image_files)}张MRI图像，将为每张生成{num_images_per_original}张形变图像")

    for img_file in tqdm(image_files, desc="处理MRI图像"):
        # 读取原始图像
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        base_name = os.path.splitext(img_file)[0]

        # 为每张原始图像生成多张形变图像
        for i in range(num_images_per_original):
            # 随机决定是否应用弹性形变
            use_elastic = random.random() > 0.2  # 80%的概率使用弹性形变

            # 随机决定是否应用仿射变换
            use_affine = random.random() > 0.2  # 80%的概率使用仿射变换

            deformed_img = img.copy()

            # 应用仿射变换
            if use_affine:
                max_rotation = random.uniform(affine_rotation_range[0], affine_rotation_range[1])
                max_translation = random.uniform(affine_translation_range[0], affine_translation_range[1])
                max_scale = random.uniform(affine_scale_range[0], affine_scale_range[1])

                deformed_img = affine_transform(
                    deformed_img,
                    max_rotation=max_rotation,
                    max_translation=max_translation,
                    max_scale=max_scale
                )

            # 应用弹性形变
            if use_elastic:
                alpha = random.uniform(elastic_alpha_range[0], elastic_alpha_range[1])
                sigma = random.uniform(elastic_sigma_range[0], elastic_sigma_range[1])

                deformed_img = elastic_transform(
                    deformed_img,
                    alpha=alpha,
                    sigma=sigma,
                    random_state=np.random.RandomState(None)
                )

            # 保存形变后的图像
            output_path = os.path.join(output_folder, f"{base_name}_deformed_{i + 1}.png")
            cv2.imwrite(output_path, deformed_img)


def visualize_deformation(original_image_path, output_folder, num_examples=3):
    """
    可视化原始图像和形变后的图像进行对比

    参数:
    - original_image_path: 原始图像的路径
    - output_folder: 包含形变图像的文件夹
    - num_examples: 要显示的形变示例数量
    """
    # 读取原始图像
    original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        print(f"无法读取原始图像: {original_image_path}")
        return

    base_name = os.path.splitext(os.path.basename(original_image_path))[0]

    # 查找对应的形变图像
    deformed_files = [f for f in os.listdir(output_folder) if f.startswith(base_name) and "_deformed_" in f]

    if not deformed_files:
        print(f"未找到与{base_name}相关的形变图像")
        return

    # 限制示例数量
    num_examples = min(num_examples, len(deformed_files))
    selected_files = random.sample(deformed_files, num_examples)

    # 创建图形
    fig, axes = plt.subplots(1, num_examples + 1, figsize=(4 * (num_examples + 1), 4))

    # 显示原始图像
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("原始图像")
    axes[0].axis('off')

    # 显示形变图像
    for i, file in enumerate(selected_files):
        deformed_path = os.path.join(output_folder, file)
        deformed_img = cv2.imread(deformed_path, cv2.IMREAD_GRAYSCALE)

        axes[i + 1].imshow(deformed_img, cmap='gray')
        axes[i + 1].set_title(f"形变示例 {i + 1}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{base_name}_comparison.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    # 设置输入和输出文件夹
    input_folder = "data/test/MRI"
    output_folder = "data/test/MRI_deformed"

    # 生成形变图像
    generate_deformed_images(
        input_folder=input_folder,
        output_folder=output_folder,
        num_images_per_original=5,  # 每张原图生成5张形变图像
        elastic_alpha_range=(10, 50),  # 显著降低弹性形变强度
        elastic_sigma_range=(12, 16),  # 增加平滑程度
        affine_rotation_range=(1, 2),  # 减小旋转角度
        affine_translation_range=(1, 2),  # 减小平移距离
        affine_scale_range=(0.01, 0.02)  # 减小缩放范围
    )

    # 可视化一个示例图像的形变效果（可选）
    # 假设第一张图像为示例
    sample_images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if sample_images:
        visualize_deformation(
            original_image_path=os.path.join(input_folder, sample_images[0]),
            output_folder=output_folder,
            num_examples=3
        )
