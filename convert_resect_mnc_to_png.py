"""
RESECT 数据集 MNC → PNG 转换脚本
=================================
将 RESECT-MINC-14-27 文件夹中的 .mnc / .mnc.gz 3D 医学影像
逐层切片并保存为 PNG 图像。

输出目录结构:
  resect_png/
    Case14/
      MRI_T1/        → 0.png, 1.png, ...
      MRI_FLAIR/     → 0.png, 1.png, ...
      US_before/     → 0.png, 1.png, ...
      US_after/      → 0.png, 1.png, ...
      US_during/     → 0.png, 1.png, ...
    Case15/
      ...
"""

import os
import gzip
import shutil
import tempfile
import numpy as np
from PIL import Image
import nibabel as nib
from tqdm import tqdm


# ============ 配置 ============
INPUT_DIR  = "./RESECT-MINC-14-27"
OUTPUT_DIR = "./resect_png"
# 切片方向: 'axial'(沿最后一个维度), 'all'(三个方向都切)
SLICE_AXIS = 'axial'
# 是否跳过大量黑色的切片 (有效像素 < 阈值)
SKIP_EMPTY = True
EMPTY_THRESHOLD = 0.02   # 少于2%有效像素的切片跳过


def decompress_gz(gz_path, tmp_dir):
    """解压 .mnc.gz → 临时 .mnc 文件"""
    basename = os.path.basename(gz_path).replace('.gz', '')
    out_path = os.path.join(tmp_dir, basename)
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_path


def normalize_to_uint8(volume):
    """将体数据归一化到 [0, 255] uint8"""
    v = volume.astype(np.float64)
    vmin, vmax = np.percentile(v[v > 0], 1) if (v > 0).any() else 0, np.percentile(v, 99.5)
    if vmax <= vmin:
        vmax = v.max()
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.uint8)
    v = np.clip((v - vmin) / (vmax - vmin), 0, 1)
    return (v * 255).astype(np.uint8)


def save_slices(volume, output_dir, axis='axial', skip_empty=True, threshold=0.02):
    """
    沿指定轴切片并保存为 PNG

    参数:
        volume: 3D numpy array (已归一化到 uint8)
        output_dir: 输出目录
        axis: 'axial' (最后一维), 'coronal' (中间维), 'sagittal' (第一维)
        skip_empty: 是否跳过空切片
        threshold: 有效像素比例阈值
    返回:
        保存的切片数量
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    if axis == 'axial':
        n_slices = volume.shape[2]
        get_slice = lambda i: volume[:, :, i]
    elif axis == 'coronal':
        n_slices = volume.shape[1]
        get_slice = lambda i: volume[:, i, :]
    elif axis == 'sagittal':
        n_slices = volume.shape[0]
        get_slice = lambda i: volume[i, :, :]
    else:
        raise ValueError(f"未知轴: {axis}")

    for i in range(n_slices):
        s = get_slice(i)

        # 检查是否空切片
        if skip_empty:
            content_ratio = (s > 10).sum() / s.size
            if content_ratio < threshold:
                continue

        # 保存
        img = Image.fromarray(s)
        img.save(os.path.join(output_dir, f"{i}.png"))
        saved += 1

    return saved


def find_mnc_file(directory, pattern):
    """
    查找 mnc 文件，优先用未压缩的 .mnc，否则用 .mnc.gz

    返回:
        (file_path, is_compressed)
    """
    # 先找 .mnc
    for f in os.listdir(directory):
        if f.endswith('.mnc') and not f.endswith('.mnc.gz') and pattern.lower() in f.lower():
            return os.path.join(directory, f), False

    # 再找 .mnc.gz
    for f in os.listdir(directory):
        if f.endswith('.mnc.gz') and pattern.lower() in f.lower():
            return os.path.join(directory, f), True

    return None, False


def process_case(case_dir, output_base, tmp_dir):
    """处理单个病人"""
    case_name = os.path.basename(case_dir)
    case_output = os.path.join(output_base, case_name)

    mri_dir = os.path.join(case_dir, 'MRI')
    us_dir  = os.path.join(case_dir, 'US')

    # 定义要转换的文件
    targets = []
    if os.path.isdir(mri_dir):
        targets += [
            (mri_dir, 'T1',     'MRI_T1'),
            (mri_dir, 'FLAIR',  'MRI_FLAIR'),
        ]
    if os.path.isdir(us_dir):
        targets += [
            (us_dir, 'US-before',  'US_before'),
            (us_dir, 'US-after',   'US_after'),
            (us_dir, 'US-during',  'US_during'),
        ]

    results = {}
    for src_dir, pattern, out_name in targets:
        fpath, is_gz = find_mnc_file(src_dir, pattern)
        if fpath is None:
            print(f"    [跳过] {case_name}/{out_name}: 未找到匹配 '{pattern}' 的文件")
            continue

        # 解压缩（如需）
        if is_gz:
            print(f"    解压 {os.path.basename(fpath)}...")
            actual_path = decompress_gz(fpath, tmp_dir)
        else:
            actual_path = fpath

        # 读取
        try:
            img = nib.load(actual_path)
            data = img.get_fdata()
        except Exception as e:
            print(f"    [错误] {case_name}/{out_name}: 读取失败 - {e}")
            continue

        # 归一化
        vol_uint8 = normalize_to_uint8(data)

        # 切片并保存
        out_dir = os.path.join(case_output, out_name)
        n_saved = save_slices(vol_uint8, out_dir,
                              axis=SLICE_AXIS,
                              skip_empty=SKIP_EMPTY,
                              threshold=EMPTY_THRESHOLD)

        results[out_name] = {
            'shape': data.shape,
            'saved_slices': n_saved,
            'total_slices': data.shape[2] if SLICE_AXIS == 'axial' else data.shape[1 if SLICE_AXIS == 'coronal' else 0],
        }

        print(f"    ✓ {out_name}: shape={data.shape}, "
              f"保存 {n_saved}/{results[out_name]['total_slices']} 切片 → {out_dir}")

        # 清理解压的临时文件
        if is_gz and os.path.exists(actual_path):
            os.remove(actual_path)

    return results


def main():
    print("=" * 60)
    print("RESECT 数据集 MNC → PNG 转换")
    print("=" * 60)
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  切片方向: {SLICE_AXIS}")
    print(f"  跳过空切片: {SKIP_EMPTY} (阈值 {EMPTY_THRESHOLD*100:.0f}%)")
    print()

    if not os.path.isdir(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        return

    # 找到所有 Case 目录
    cases = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d)) and d.startswith('Case')
    ])

    print(f"找到 {len(cases)} 个病例: {', '.join(cases)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 使用临时目录存放解压文件
    with tempfile.TemporaryDirectory() as tmp_dir:
        all_results = {}
        for case in cases:
            case_dir = os.path.join(INPUT_DIR, case)
            print(f"[{case}]")
            results = process_case(case_dir, OUTPUT_DIR, tmp_dir)
            all_results[case] = results
            print()

    # 统计汇总
    print("=" * 60)
    print("转换完成! 统计汇总:")
    print("=" * 60)

    total_slices = 0
    for case, res in all_results.items():
        case_total = sum(r['saved_slices'] for r in res.values())
        total_slices += case_total
        modalities = ', '.join(f"{k}({v['saved_slices']})" for k, v in res.items())
        print(f"  {case}: {case_total} 切片  [{modalities}]")

    print(f"\n  总计: {total_slices} 个 PNG 切片")
    print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
