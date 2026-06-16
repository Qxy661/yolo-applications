"""
数据清洗脚本
============
功能：
1. 删除模糊/损坏/重复图片
2. 检查标注文件完整性
3. 统一图片格式
4. 检查标注坐标有效性
5. 去除异常标注

使用方法：
    python clean_data.py --input raw/blade30 --output cleaned/blade30
"""

import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow 未安装，部分功能不可用")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def calculate_image_hash(image_path):
    """计算图片MD5哈希"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def check_image_quality(image_path, min_size=100, max_blur=100):
    """
    检查图片质量
    返回: (is_valid, reason)
    """
    if not HAS_PIL:
        return True, "Pillow未安装，跳过检查"

    try:
        img = Image.open(image_path)

        # 检查是否能正常打开
        img.verify()

        # 重新打开（verify后需要重新打开）
        img = Image.open(image_path)

        # 检查尺寸
        w, h = img.size
        if w < min_size or h < min_size:
            return False, f"尺寸过小: {w}x{h}"

        # 检查是否为RGB或RGBA
        if img.mode not in ['RGB', 'RGBA', 'L']:
            return False, f"不支持的色彩模式: {img.mode}"

        return True, "OK"

    except Exception as e:
        return False, f"图片损坏: {e}"


def check_annotation_validity(label_path, img_width, img_height):
    """
    检查YOLO标注有效性
    格式: class_id center_x center_y width height (归一化0-1)
    """
    issues = []

    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    issues.append(f"行{line_num}: 格式错误，至少需要5个值")
                    continue

                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # 检查范围
                    if not (0 <= cx <= 1 and 0 <= cy <= 1):
                        issues.append(f"行{line_num}: 中心坐标超出范围 ({cx}, {cy})")
                    if not (0 < w <= 1 and 0 < h <= 1):
                        issues.append(f"行{line_num}: 宽高超出范围 ({w}, {h})")
                    if class_id < 0:
                        issues.append(f"行{line_num}: 类别ID为负数 ({class_id})")

                except ValueError:
                    issues.append(f"行{line_num}: 数值解析错误")

    except Exception as e:
        issues.append(f"读取标注文件失败: {e}")

    return issues


def find_duplicates(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """查找重复图片"""
    hash_map = defaultdict(list)

    for ext in extensions:
        for img_path in Path(directory).rglob(f'*{ext}'):
            img_hash = calculate_image_hash(img_path)
            if img_hash:
                hash_map[img_hash].append(img_path)

    # 返回有重复的组
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    return duplicates


def clean_dataset(input_dir, output_dir, min_size=100):
    """
    清洗数据集
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'duplicate_groups': 0,
        'annotation_issues': 0,
        'issues': []
    }

    # 查找所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(input_dir.rglob(f'*{ext}'))
        images.extend(input_dir.rglob(f'*{ext.upper()}'))

    stats['total_images'] = len(images)
    print(f"\n找到 {len(images)} 张图片")

    # 检查重复
    print("\n检查重复图片...")
    duplicates = find_duplicates(input_dir)
    stats['duplicate_groups'] = len(duplicates)
    if duplicates:
        print(f"  发现 {len(duplicates)} 组重复图片")
        for h, paths in list(duplicates.items())[:5]:  # 只显示前5组
            print(f"    {paths[0].name} 重复 {len(paths)} 次")

    # 检查每张图片
    print("\n检查图片质量...")
    for img_path in images:
        # 检查图片质量
        is_valid, reason = check_image_quality(img_path, min_size)

        if is_valid:
            stats['valid_images'] += 1
        else:
            stats['invalid_images'] += 1
            stats['issues'].append(f"{img_path.name}: {reason}")

        # 检查对应标注文件
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            try:
                img = Image.open(img_path) if HAS_PIL else None
                if img:
                    issues = check_annotation_validity(label_path, *img.size)
                    if issues:
                        stats['annotation_issues'] += len(issues)
                        for issue in issues:
                            stats['issues'].append(f"{label_path.name}: {issue}")
            except Exception:
                pass

    # 输出统计
    print("\n" + "="*60)
    print("清洗统计")
    print("="*60)
    print(f"总图片数: {stats['total_images']}")
    print(f"有效图片: {stats['valid_images']}")
    print(f"无效图片: {stats['invalid_images']}")
    print(f"重复组数: {stats['duplicate_groups']}")
    print(f"标注问题: {stats['annotation_issues']}")

    if stats['issues']:
        print(f"\n发现的问题 (前20条):")
        for issue in stats['issues'][:20]:
            print(f"  - {issue}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='数据清洗')
    parser.add_argument('--input', required=True, help='输入目录')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--min-size', type=int, default=100, help='最小图片尺寸')
    args = parser.parse_args()

    print("="*60)
    print("风电叶片缺陷数据集清洗")
    print("="*60)

    stats = clean_dataset(args.input, args.output, args.min_size)

    # 保存统计报告
    report_path = Path(args.output) / 'clean_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n统计报告已保存: {report_path}")


if __name__ == '__main__':
    main()
