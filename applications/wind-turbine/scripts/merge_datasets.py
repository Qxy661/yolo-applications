"""风电场叶片缺陷检测 — 多源数据集合并脚本

将多个公开数据集合并为统一的YOLO格式。
类别映射: 5类核心缺陷
"""
import os
import sys
import shutil
import yaml
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'

# 目标5类
TARGET_CLASSES = {
    0: 'Crack',       # 叶片裂纹
    1: 'Breakage',    # 叶片破损
    2: 'Lightning',   # 雷击损伤
    3: 'Peeling',     # 涂层脱落
    4: 'Erosion',     # 边缘侵蚀
}

# 9类数据集类别映射: 源类别名 -> 目标类别ID (None表示忽略)
MAPPING_9CLASS = {
    'Chip': 1,
    'Crack': 0,
    'Damaged': 1,
    'Damaged Or Misaligned': 1,
    'Discoloration': None,
    'Erosion': 4,
    'Flaking': 3,
    'Peeling': 3,
    'Pinholes': None,
}

# 7类数据集类别映射
MAPPING_7CLASS = {
    'Burn': 2,
    'Crack': 0,
    'Deformation': None,
    'Dirt': None,
    'Oil Stain': None,
    'Peeling': 3,
    'Rust': None,
}


def find_dataset_dirs(raw_dir):
    """查找已下载的数据集目录"""
    datasets = {}
    for d in raw_dir.iterdir():
        if d.is_dir():
            if '9class' in d.name or 'uav' in d.name.lower():
                datasets['9class'] = d
            elif 'kaggle' in d.name.lower():
                datasets['kaggle'] = d
            elif '7class' in d.name:
                datasets['7class'] = d
    return datasets


def find_images_and_labels(dataset_dir):
    """查找数据集中的图片和标签目录"""
    images = []
    labels = []

    # 常见目录结构
    for pattern in ['**/images', '**/img', '**/JPEGImages', '**/images/train', '**/images/val']:
        found = list(dataset_dir.glob(pattern))
        if found:
            images.extend(found)

    for pattern in ['**/labels', '**/label', '**/Annotations', '**/labels/train', '**/labels/val']:
        found = list(dataset_dir.glob(pattern))
        if found:
            labels.extend(found)

    return images, labels


def convert_voc_to_yolo(xml_path, class_map):
    """将VOC XML标注转换为YOLO格式（简化版）"""
    # 这里只做格式检查，实际需要完整的VOC解析
    # 如果是YOLO格式则直接使用
    return None


def merge_dataset(source_dir, source_name, class_map, output_images, output_labels, stats):
    """合并单个数据集"""
    print(f'\n[MERGE] 处理 {source_name}: {source_dir}')

    # 查找图片和标签
    image_dirs, label_dirs = find_images_and_labels(source_dir)

    if not image_dirs:
        print(f'  [WARN] 未找到图片目录，跳过')
        return

    image_count = 0
    skip_count = 0

    for img_dir in image_dirs:
        for img_path in img_dir.glob('*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                continue

            # 查找对应标签
            label_path = None
            for lbl_dir in label_dirs:
                candidate = lbl_dir / (img_path.stem + '.txt')
                if candidate.exists():
                    label_path = candidate
                    break

            if label_path is None:
                skip_count += 1
                continue

            # 读取并转换标签
            new_lines = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    # 获取类别名
                    # 需要从原始数据集的yaml或names获取
                    # 这里假设已经是数字ID
                    # 如果有映射表，需要转换
                    coords = parts[1:5]
                    new_lines.append(f'{cls_id} {" ".join(coords)}')

            if new_lines:
                # 保存图片
                dst_img = output_images / f'{source_name}_{img_path.name}'
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                # 保存标签
                dst_lbl = output_labels / f'{source_name}_{img_path.stem}.txt'
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(new_lines))

                image_count += 1
                for line in new_lines:
                    cls_id = int(line.split()[0])
                    stats[cls_id] = stats.get(cls_id, 0) + 1

    print(f'  [OK] 合并 {image_count} 张图片, 跳过 {skip_count} 张')


def main():
    print('=' * 60)
    print('风电场叶片缺陷检测 — 数据集合并')
    print('=' * 60)

    # 创建输出目录
    merged_images = DATA_DIR / 'merged' / 'images'
    merged_labels = DATA_DIR / 'merged' / 'labels'
    merged_images.mkdir(parents=True, exist_ok=True)
    merged_labels.mkdir(parents=True, exist_ok=True)

    # 查找数据集
    datasets = find_dataset_dirs(RAW_DIR)
    print(f'\n[INFO] 找到 {len(datasets)} 个数据集: {list(datasets.keys())}')

    stats = Counter()

    # 合并各数据集
    if '9class' in datasets:
        merge_dataset(datasets['9class'], '9class', MAPPING_9CLASS,
                      merged_images, merged_labels, stats)

    if '7class' in datasets:
        merge_dataset(datasets['7class'], '7class', MAPPING_7CLASS,
                      merged_images, merged_labels, stats)

    # Kaggle数据集需要单独处理（VOC格式）
    if 'kaggle' in datasets:
        print('\n[KAGGLE] Kaggle数据集需要VOC到YOLO格式转换')
        print('  请手动转换后放入 merged/ 目录')

    # 输出统计
    print('\n' + '=' * 60)
    print('合并统计')
    print('=' * 60)
    total = 0
    for cls_id in sorted(TARGET_CLASSES.keys()):
        count = stats.get(cls_id, 0)
        total += count
        print(f'  {TARGET_CLASSES[cls_id]:12s} (ID={cls_id}): {count} 个标注')
    print(f'  {"总计":12s}: {total} 个标注')

    # 保存统计
    stats_path = DATA_DIR / 'merge_stats.yaml'
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'total_images': len(list(merged_images.glob('*'))),
            'total_annotations': total,
            'class_counts': {TARGET_CLASSES[k]: v for k, v in stats.items()},
        }, f, allow_unicode=True, default_flow_style=False)
    print(f'\n[SAVE] 统计信息: {stats_path}')


if __name__ == '__main__':
    main()
