"""
数据集划分脚本
============
功能：
1. 按比例划分训练/验证/测试集 (默认 7:1:2)
2. 分层抽样，确保各类别分布一致
3. 生成划分清单文件
4. 复制文件到标准YOLO目录结构

使用方法：
    python split_dataset.py --input merged --output processed/detection --ratios 0.7 0.1 0.2
"""

import os
import sys
import argparse
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# 标准5类
STANDARD_CLASSES = {
    'crack': 0,
    'erosion': 1,
    'lightning': 2,
    'peeling': 3,
    'hole': 4,
}


def read_labels(label_path):
    """读取标注文件，返回类别ID列表"""
    classes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.append(int(parts[0]))
    except Exception:
        pass
    return classes


def stratified_split(image_list, label_map, ratios, seed=42):
    """
    分层抽样划分
    确保每个类别的图片在各子集中的比例一致
    """
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    random.seed(seed)

    # 按主要类别分组（取每张图片中类别ID最小的作为主要类别）
    class_groups = defaultdict(list)
    no_label = []

    for img_path in image_list:
        label_path = label_map.get(str(img_path))
        if label_path and label_path.exists():
            classes = read_labels(label_path)
            if classes:
                # 用最小类别ID作为分组键
                primary_class = min(classes)
                class_groups[primary_class].append(img_path)
            else:
                no_label.append(img_path)
        else:
            no_label.append(img_path)

    train, val, test = [], [], []

    # 对每个类别分别划分
    for class_id, images in class_groups.items():
        random.shuffle(images)
        n = len(images)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # 确保test也有
        if n >= 3:
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = n - n_train - n_val
        else:
            n_train = n
            n_val = 0
            n_test = 0

        train.extend(images[:n_train])
        val.extend(images[n_train:n_train + n_val])
        test.extend(images[n_train + n_val:])

    # 无标注的图片也划分
    if no_label:
        random.shuffle(no_label)
        n = len(no_label)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(no_label[:n_train])
        val.extend(no_label[n_train:n_train + n_val])
        test.extend(no_label[n_train + n_val:])

    return train, val, test


def split_dataset(input_dir, output_dir, ratios=(0.7, 0.1, 0.2), seed=42):
    """
    划分数据集
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 查找图片和标注
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    for ext in image_exts:
        images.extend(input_dir.glob(f'images/*{ext}'))
        images.extend(input_dir.glob(f'images/*{ext.upper()}'))

    label_map = {}
    for img in images:
        label_path = input_dir / 'labels' / (img.stem + '.txt')
        if label_path.exists():
            label_map[str(img)] = label_path

    print(f"\n找到 {len(images)} 张图片, {len(label_map)} 个标注文件")

    if not images:
        print("[ERROR] 未找到图片文件")
        return None

    # 分层抽样划分
    train_imgs, val_imgs, test_imgs = stratified_split(images, label_map, ratios, seed)

    stats = {
        'train': len(train_imgs),
        'val': len(val_imgs),
        'test': len(test_imgs),
        'total': len(train_imgs) + len(val_imgs) + len(test_imgs),
        'ratios': {'train': ratios[0], 'val': ratios[1], 'test': ratios[2]},
        'class_distribution': {
            'train': Counter(),
            'val': Counter(),
            'test': Counter(),
        },
    }

    # 创建目录并复制文件
    for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        out_img_dir = output_dir / 'images' / split_name
        out_label_dir = output_dir / 'labels' / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            # 复制图片
            dst_img = out_img_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            # 复制标注
            label_path = label_map.get(str(img_path))
            if label_path:
                dst_label = out_label_dir / (img_path.stem + '.txt')
                shutil.copy2(label_path, dst_label)

                # 统计类别
                classes = read_labels(label_path)
                for c in classes:
                    cname = [k for k, v in STANDARD_CLASSES.items() if v == c][0]
                    stats['class_distribution'][split_name][cname] += 1

        print(f"  {split_name}: {len(split_imgs)} 张")

    # 转换Counter为dict
    for split in stats['class_distribution']:
        stats['class_distribution'][split] = dict(stats['class_distribution'][split])

    # 输出统计
    print("\n" + "=" * 60)
    print("划分统计")
    print("=" * 60)
    print(f"训练集: {stats['train']} 张 ({stats['train']/stats['total']*100:.1f}%)")
    print(f"验证集: {stats['val']} 张 ({stats['val']/stats['total']*100:.1f}%)")
    print(f"测试集: {stats['test']} 张 ({stats['test']/stats['total']*100:.1f}%)")

    print("\n各类别分布:")
    for cls_name in sorted(STANDARD_CLASSES.keys()):
        train_n = stats['class_distribution']['train'].get(cls_name, 0)
        val_n = stats['class_distribution']['val'].get(cls_name, 0)
        test_n = stats['class_distribution']['test'].get(cls_name, 0)
        total_n = train_n + val_n + test_n
        print(f"  {cls_name}: {total_n} (train={train_n}, val={val_n}, test={test_n})")

    return stats


def main():
    parser = argparse.ArgumentParser(description='划分数据集')
    parser.add_argument('--input', required=True, help='合并后的数据目录')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--ratios', nargs=3, type=float, default=[0.7, 0.1, 0.2],
                        help='训练:验证:测试比例 (默认 0.7 0.1 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    print("=" * 60)
    print("风电叶片缺陷数据集划分")
    print("=" * 60)

    stats = split_dataset(args.input, args.output, args.ratios, args.seed)

    if stats:
        report_path = Path(args.output) / 'split_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n统计报告已保存: {report_path}")


if __name__ == '__main__':
    main()
