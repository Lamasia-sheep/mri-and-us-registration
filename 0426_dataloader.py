import os
import glob
import numpy as np
from PIL import Image
import re
import sys
import traceback

# 尝试导入医学图像处理库
try:
    import nibabel as nib

    has_nibabel = True
    print("使用nibabel库处理医学图像")
except ImportError:
    has_nibabel = False
    print("nibabel库未安装")

try:
    import SimpleITK as sitk

    has_sitk = True
    print("使用SimpleITK库处理医学图像")
except ImportError:
    has_sitk = False
    print("SimpleITK库未安装")

if not has_nibabel and not has_sitk:
    print("错误: 需要安装nibabel或SimpleITK库")
    print("运行: pip install nibabel SimpleITK")
    sys.exit(1)

# 配置
BASE_DIR = "group2"  # 您的group2目录路径
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")
MAX_FILES = 20  # 限制处理的文件数量

# 要处理的患者ID列表（可以扩展）
PATIENT_IDS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']  # 先从患者10开始，成功后可以扩展为["01", "02", ..., "14"]


def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_slice_number(filename):
    """从2D超声文件名中提取切片编号"""
    match = re.search(r'\.(\d+)sm\.', filename)
    if match:
        return int(match.group(1))
    return None


def read_medical_image(file_path):
    """
    读取医学图像文件（尝试不同的库）
    """
    print(f"正在读取: {file_path}")
    try:
        if has_nibabel:
            try:
                # 尝试使用nibabel读取
                img = nib.load(file_path)
                data = img.get_fdata()
                print(f"使用nibabel成功读取，数据形状: {data.shape}")
                return data
            except Exception as e:
                print(f"nibabel读取失败: {e}")
                traceback.print_exc()

        if has_sitk:
            try:
                # 尝试使用SimpleITK读取
                img = sitk.ReadImage(file_path)
                data = sitk.GetArrayFromImage(img)
                # SimpleITK可能会转置维度，根据需要调整
                if len(data.shape) == 3:
                    data = np.transpose(data, (2, 1, 0))
                print(f"使用SimpleITK成功读取，数据形状: {data.shape}")
                return data
            except Exception as e:
                print(f"SimpleITK读取失败: {e}")
                traceback.print_exc()

        print(f"所有库都无法读取文件: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        traceback.print_exc()
        return None


def save_as_png(data, output_file, slice_idx=None):
    """将NumPy数组保存为PNG，可选择提取特定切片"""
    try:
        # 如果需要，提取切片
        if slice_idx is not None and len(data.shape) == 3:
            # 确保slice_idx在有效范围内
            if slice_idx < 0:
                slice_idx = 0
            if slice_idx >= data.shape[2]:
                slice_idx = data.shape[2] - 1

            data = data[:, :, slice_idx]
            print(f"提取切片 {slice_idx}，形状: {data.shape}")

        # 归一化为0-255
        min_val = np.min(data)
        max_val = np.max(data)
        print(f"数据范围: {min_val} 到 {max_val}")

        if max_val > min_val:
            normalized = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            print("警告: 数据无值范围，全设为0")
            normalized = np.zeros(data.shape, dtype=np.uint8)

        # 确保数据是2D的
        if len(normalized.shape) > 2:
            print(f"警告: 数据维度 > 2, 形状: {normalized.shape}，取第一层")
            normalized = normalized[:, :, 0]

        # 增强对比度（可选）
        p2, p98 = np.percentile(normalized, (2, 98))
        if p98 > p2:
            normalized = np.clip(normalized, p2, p98)
            normalized = ((normalized - p2) / (p98 - p2) * 255).astype(np.uint8)

        # 保存为PNG
        Image.fromarray(normalized).save(output_file)
        print(f"图像已保存至 {output_file}")
        return True
    except Exception as e:
        print(f"保存PNG时出错: {e}")
        traceback.print_exc()
        return False


def process_patient(patient_id):
    """处理单个患者的数据"""
    print(f"\n开始处理患者 {patient_id}...")

    # 创建该患者的输出目录
    patient_output_dir = os.path.join(OUTPUT_BASE_DIR, f"patient_{patient_id}")
    mri_output_dir = os.path.join(patient_output_dir, "mri_png")
    ensure_dir(mri_output_dir)

    patient_dir = os.path.join(BASE_DIR, patient_id)
    if not os.path.exists(patient_dir):
        print(f"未找到患者 {patient_id} 的目录: {patient_dir}")
        return False

    # 查找MRI文件
    mri_file = os.path.join(patient_dir, f"{patient_id}_mr_tal.mnc")
    if not os.path.exists(mri_file):
        print(f"未找到MRI文件: {mri_file}")
        return False
    print(f"找到MRI文件: {mri_file}")

    # 查找超声文件目录
    us_dir = os.path.join(patient_dir, "2dus")
    if not os.path.exists(us_dir):
        print(f"未找到2dus目录: {us_dir}")
        return False

    # 获取所有2D超声文件（用于获取切片编号）
    us_pattern = os.path.join(us_dir, f"{patient_id}*.2dus.*.mnc")
    us_files = glob.glob(us_pattern)
    if not us_files:
        # 尝试不同的模式
        us_pattern = os.path.join(us_dir, "*.2dus.*.mnc")
        us_files = glob.glob(us_pattern)
        if not us_files:
            print(f"未找到2D超声文件，模式: {us_pattern}")
            print(f"2dus目录中的文件: {os.listdir(us_dir)}")
            return False

    print(f"找到 {len(us_files)} 个2D超声文件，用于确定切片编号")

    # 加载MRI数据
    print("正在加载MRI数据...")
    mri_data = read_medical_image(mri_file)
    if mri_data is None:
        print("加载MRI数据失败")
        return False

    print(f"MRI数据形状: {mri_data.shape}")

    # 保存MRI中间切片作为参考
    if len(mri_data.shape) == 3:
        mid_slice = mri_data.shape[2] // 2
        mri_mid_png = os.path.join(mri_output_dir, f"{patient_id}_mid_slice.png")
        if not save_as_png(mri_data, mri_mid_png, slice_idx=mid_slice):
            print("保存MRI中间切片失败")

    # 处理与超声对应的MRI切片
    processed_count = 0
    for i, us_file in enumerate(us_files[:MAX_FILES]):
        # 从文件名中提取切片编号
        slice_num = extract_slice_number(us_file)
        if slice_num is None:
            print(f"无法从 {os.path.basename(us_file)} 中提取切片编号")
            continue

        print(f"处理切片编号: {slice_num}")

        # 计算对应的MRI切片编号
        if len(mri_data.shape) == 3:
            # 线性映射切片编号到MRI卷
            mri_slice_idx = min(int((slice_num / 1000) * mri_data.shape[2]), mri_data.shape[2] - 1)

            # 保存MRI切片为PNG
            mri_png = os.path.join(mri_output_dir, f"{patient_id}_slice_{slice_num:05d}.png")
            if save_as_png(mri_data, mri_png, slice_idx=mri_slice_idx):
                processed_count += 1
                print(f"成功保存MRI切片: 编号 {slice_num}")
        else:
            print(f"MRI数据不是3D的，无法提取切片")

    print(f"患者 {patient_id} 处理完成，成功处理 {processed_count} 个切片")
    return processed_count > 0


def main():
    # 创建主输出目录
    print("创建输出目录...")
    ensure_dir(OUTPUT_BASE_DIR)

    # 处理所有指定的患者
    success_count = 0
    for patient_id in PATIENT_IDS:
        if process_patient(patient_id):
            success_count += 1

    print(f"总共处理了 {len(PATIENT_IDS)} 个患者，成功 {success_count} 个")


if __name__ == "__main__":
    main()