"""
数据集合并脚本
============
功能：
1. 合并多个数据集到统一目录
2. 基于图片哈希去重
3. 统一类别映射
4. 生成合并统计报告

使用方法：
    python merge_datasets.py --input raw/blade30 raw/qq767172261_6cls raw/qq767172261_uav5 --output merged
"""

import os
import sys
import argparse
import json
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# 标准5类映射
STANDARD_CLASSES = {
    'crack': 0,
    'erosion': 1,
    'lightning': 2,
    'peeling': 3,
    'hole': 4,
}

# 各数据集类别映射（源类别 → 标准类别）
DATASET_CLASS_MAP = {
    'blade30': {
        'crack': 'crack',
        'erosion': 'erosion',
        'contamination': 'erosion',
        'pollution': 'erosion',
        'lightning': 'lightning',
        'peeling': 'peeling',
        'hole': 'hole',
    },
    'qq767172261_6cls': {
        'crack': 'crack',
        'erosion': 'erosion',
        'dirt': 'erosion',
        'oil leakage': 'erosion',
        'oil_leakage': 'erosion',
        'pu-tape': 'peeling',
        'pu_tape': 'peeling',
        'pin hole': 'hole',
        'pin_hole': 'hole',
    },
    'qq767172261_uav5': {
        'oil leakage': 'erosion',
        'oil_leakage': 'erosion',
        'dirt': 'erosion',
        'paint': 'peeling',
        'le-erosion': 'erosion',
        'le_erosion': 'erosion',
        'pu-tape': 'peeling',
        'pu_tape': 'peeling',
    },
}

# 中文映射（兜底）
CHINESE_CLASS_MAP = {
    '裂纹': 'crack',
    '侵蚀': 'erosion',
    '雷击': 'lightning',
    '涂层脱落': 'peeling',
    '涂层损伤': 'peeling',
    '剥落': 'peeling',
    '漆面': 'peeling',
    '孔洞': 'hole',
    '损伤': 'erosion',
}


def calculate_file_hash(file_path):
    """计算文件MD5哈希"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def detect_dataset_name(path):
    """根据路径推断数据集名称"""
    path_str = str(path).lower()
    if 'blade30' in path_str:
        return 'blade30'
    elif 'qq' in path_str and ('6' in path_str or 'cls' in path_str):
        return 'qq767172261_6cls'
    elif 'qq' in path_str and ('5' in path_str or 'uav' in path_str):
        return 'qq767172261_uav5'
    elif 'wtbd' in path_str:
        return 'wtbd'
    else:
        return 'unknown'


def find_images_and_labels(data_dir):
    """查找目录中的图片和标注文件"""
    data_dir = Path(data_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    labels = []

    for ext in image_exts:
        images.extend(data_dir.rglob(f'*{ext}'))
        images.extend(data_dir.rglob(f'*{ext.upper()}'))

    # 查找YOLO txt标注
    for txt in data_dir.rglob('*.txt'):
        # 排除README等非标注文件
        if txt.name.lower() in ('readme.txt', 'classes.txt', 'notes.txt'):
            continue
        # 检查是否是YOLO格式（第一行应该是数字开头）
        try:
            with open(txt, 'r') as f:
                first_line = f.readline().strip()
                if first_line and first_line[0].isdigit():
                    labels.append(txt)
        except Exception:
            pass

    return images, labels


def remap_labels(label_path, dataset_name, output_path):
    """重新映射标注类别"""
    class_map = DATASET_CLASS_MAP.get(dataset_name, {})
    lines = []
    skipped = 0

    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                # 反向查找原始类别名
                orig_name = None
                for name, cid in STANDARD_CLASSES.items():
                    if cid == class_id:
                        orig_name = name
                        break

                if orig_name is None:
                    # 尝试通过class_map查找
                    mapped = class_map.get(str(class_id), None)
                    if mapped and mapped in STANDARD_CLASSES:
                        new_id = STANDARD_CLASSES[mapped]
                    else:
                        skipped += 1
                        continue
                else:
                    mapped = class_map.get(orig_name, orig_name)
                    if mapped in STANDARD_CLASSES:
                        new_id = STANDARD_CLASSES[mapped]
                    else:
                        skipped += 1
                        continue

                parts[0] = str(new_id)
                lines.append(' '.join(parts))
    except Exception as e:
        print(f"  [WARN] 读取标注失败 {label_path}: {e}")
        return 0, 0

    # 写入新标注
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(lines), skipped


def merge_datasets(input_dirs, output_dir, deduplicate=True):
    """
    合并多个数据集
    """
    output_dir = Path(output_dir)
    out_images = output_dir / 'images'
    out_labels = output_dir / 'labels'
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_images': 0,
        'total_labels': 0,
        'duplicates_removed': 0,
        'remapped': 0,
        'skipped': 0,
        'by_dataset': {},
        'class_distribution': Counter(),
    }

    seen_hashes = {}  # hash → (original_name, dataset)
    global_idx = 0

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        dataset_name = detect_dataset_name(input_dir)
        print(f"\n处理数据集: {input_dir.name} (类型: {dataset_name})")

        images, labels = find_images_and_labels(input_dir)
        print(f"  找到 {len(images)} 张图片, {len(labels)} 个标注文件")

        dataset_stats = {
            'images': len(images),
            'labels': len(labels),
            'duplicates': 0,
            'skipped': 0,
        }

        # 建立图片→标注映射
        label_map = {}
        for label in labels:
            # 查找对应的图片
            for img_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                img_candidate = label.with_suffix(img_ext)
                if img_candidate in images:
                    label_map[str(img_candidate)] = label
                    break
            # 也检查同名不同后缀
            if str(label) not in label_map.values():
                for img in images:
                    if img.stem == label.stem:
                        label_map[str(img)] = label
                        break

        for img_path in images:
            # 去重检查
            if deduplicate:
                img_hash = calculate_file_hash(img_path)
                if img_hash and img_hash in seen_hashes:
                    stats['duplicates_removed'] += 1
                    dataset_stats['duplicates'] += 1
                    continue
                if img_hash:
                    seen_hashes[img_hash] = (img_path.name, dataset_name)

            # 复制图片
            new_name = f"{dataset_name}_{global_idx:06d}{img_path.suffix}"
            dst_img = out_images / new_name
            shutil.copy2(img_path, dst_img)
            stats['total_images'] += 1
            global_idx += 1

            # 处理标注
            label_path = label_map.get(str(img_path))
            if label_path:
                dst_label = out_labels / (dst_img.stem + '.txt')
                n_lines, n_skipped = remap_labels(label_path, dataset_name, dst_label)
                stats['total_labels'] += 1
                stats['remapped'] += 1
                stats['skipped'] += n_skipped

                # 统计类别分布
                with open(dst_label, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            cname = [k for k, v in STANDARD_CLASSES.items() if v == cid][0]
                            stats['class_distribution'][cname] += 1
            else:
                dataset_stats['skipped'] += 1
                stats['skipped'] += 1

        stats['by_dataset'][dataset_name] = dataset_stats
        print(f"  处理完成: {dataset_stats['images'] - dataset_stats['duplicates']} 张有效图片")

    # 输出统计
    print("\n" + "=" * 60)
    print("合并统计")
    print("=" * 60)
    print(f"总图片数: {stats['total_images']}")
    print(f"总标注数: {stats['total_labels']}")
    print(f"去重移除: {stats['duplicates_removed']}")
    print(f"跳过: {stats['skipped']}")

    print("\n各数据集贡献:")
    for name, s in stats['by_dataset'].items():
        print(f"  {name}: {s['images'] - s['duplicates']} 张有效图片")

    if stats['class_distribution']:
        print("\n类别分布:")
        for cls, count in sorted(stats['class_distribution'].items()):
            print(f"  {cls}: {count}")

    # 转换Counter为dict以便JSON序列化
    stats['class_distribution'] = dict(stats['class_distribution'])

    return stats


def main():
    parser = argparse.ArgumentParser(description='合并风电叶片缺陷数据集')
    parser.add_argument('--input', nargs='+', required=True, help='输入目录列表')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--no-dedup', action='store_true', help='不去重')
    args = parser.parse_args()

    print("=" * 60)
    print("风电叶片缺陷数据集合并")
    print("=" * 60)

    stats = merge_datasets(
        args.input, args.output,
        deduplicate=not args.no_dedup
    )

    # 保存统计
    report_path = Path(args.output) / 'merge_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n统计报告已保存: {report_path}")


if __name__ == '__main__':
    main()
