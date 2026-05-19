"""风电场叶片缺陷检测 — 数据集划分脚本

将合并后的数据集按 8:1:1 划分为训练集/验证集/测试集。
"""
import os
import sys
import shutil
import random
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def split_dataset(merged_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """划分数据集"""
    random.seed(seed)

    images_dir = merged_dir / 'images'
    labels_dir = merged_dir / 'labels'

    # 收集所有有标签的图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    samples = []

    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in image_extensions:
            continue
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            samples.append((img_path, label_path))

    print(f'[INFO] 找到 {len(samples)} 个有效样本')

    # 打乱
    random.shuffle(samples)

    # 划分
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': samples[:n_train],
        'val': samples[n_train:n_train + n_val],
        'test': samples[n_train + n_val:],
    }

    # 复制文件
    for split_name, split_samples in splits.items():
        split_images = output_dir / 'images' / split_name
        split_labels = output_dir / 'labels' / split_name
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_samples:
            shutil.copy2(img_path, split_images / img_path.name)
            shutil.copy2(label_path, split_labels / label_path.name)

        print(f'  {split_name}: {len(split_samples)} 样本')

    return splits


def main():
    print('=' * 60)
    print('风电场叶片缺陷检测 — 数据集划分')
    print('=' * 60)

    merged_dir = DATA_DIR / 'merged'

    if not merged_dir.exists():
        print('[ERROR] 未找到合并数据集，请先运行 merge_datasets.py')
        sys.exit(1)

    # 划分
    splits = split_dataset(merged_dir, DATA_DIR)

    # 统计各类别分布
    print('\n[CLASS DISTRIBUTION]')
    class_names = {0: 'Crack', 1: 'Breakage', 2: 'Lightning', 3: 'Peeling', 4: 'Erosion'}

    for split_name in ['train', 'val', 'test']:
        labels_dir = DATA_DIR / 'labels' / split_name
        class_counts = {}
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        print(f'\n  {split_name}:')
        for cls_id in sorted(class_names.keys()):
            count = class_counts.get(cls_id, 0)
            print(f'    {class_names[cls_id]:12s}: {count}')

    print(f'\n[DONE] 数据集划分完成！')


if __name__ == '__main__':
    main()
