"""
Step 2: 完整数据处理流程
======================
功能：
1. 转换Blade30 JSON标注为YOLO格式
2. 转换WT dataset VOC XML为YOLO格式
3. 合并数据集
4. 划分训练/验证/测试集
5. 生成EDA报告

使用方法：
    python 02_process_all.py
"""

import os
import sys
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

# 配置
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'raw'
PROCESSED_DIR = BASE_DIR / 'processed'
DOCS_DIR = BASE_DIR / 'docs'

# 标准5类
STANDARD_CLASSES = {
    'crack': 0,
    'erosion': 1,
    'lightning': 2,
    'peeling': 3,
    'hole': 4,
}

# WT数据集类别映射
WT_CLASS_MAP = {
    'craze': 'crack',
    'crack': 'crack',
    'hide_craze': 'crack',
    'corrosion': 'erosion',
    'surface_injure': 'erosion',
    'thunderstrike': 'lightning',
}

# Blade30数据集类别映射 (处理分号分隔的标签)
BLADE30_CLASS_MAP = {
    'trailing edge;crack;superficial': 'crack',
    'trailing edge;crack;deep': 'crack',
    'surface;crack': 'crack',
    'leading edge;erosion;coating or LEP only': 'erosion',
    'leading edge;erosion;continuous or deep': 'erosion',
    'leading edge;erosion;spotty or laminate': 'erosion',
    'leading edge;erosion;eroded tip': 'erosion',
    'surface;contamination;other': 'erosion',
    'surface;contamination;dirt': 'erosion',
    'Surface;contamination;dirt': 'erosion',
    'surface;contamination;oil': 'erosion',
    'surface;contamination;rust': 'erosion',
    'leading edge;lightning;damage': 'lightning',
    'surface;lightning;damage': 'lightning',
    'surface;peeling': 'peeling',
    'coating;peeling': 'peeling',
    'surface;hole': 'hole',
    'leading edge;hole': 'hole',
}


def convert_blade30_json_to_yolo(json_path, img_width, img_height):
    """将Blade30 JSON标注转换为YOLO格式"""
    yolo_lines = []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])

        for shape in shapes:
            label = shape.get('label', '')

            # 尝试直接映射
            mapped_class = None
            if label.lower() in STANDARD_CLASSES:
                mapped_class = label.lower()
            elif label in BLADE30_CLASS_MAP:
                mapped_class = BLADE30_CLASS_MAP[label]
            else:
                # 尝试部分匹配
                for pattern, cls in BLADE30_CLASS_MAP.items():
                    if pattern.lower() in label.lower():
                        mapped_class = cls
                        break

            if mapped_class is None:
                continue

            class_id = STANDARD_CLASSES[mapped_class]

            # 获取点坐标
            points = shape.get('points', [])
            if len(points) < 2:
                continue

            # 转换为边界框 (xmin, ymin, xmax, ymax)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            # 转换为YOLO格式 (center_x, center_y, width, height) 归一化
            cx = (xmin + xmax) / 2 / img_width
            cy = (ymin + ymax) / 2 / img_height
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height

            # 确保在0-1范围内
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    except Exception as e:
        print(f"  [WARN] 解析JSON失败 {json_path}: {e}")

    return yolo_lines


def convert_voc_xml_to_yolo(xml_path):
    """将VOC XML标注转换为YOLO格式"""
    yolo_lines = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片尺寸
        size = root.find('size')
        if size is None:
            return yolo_lines
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 解析标注
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()

            # 映射类别
            mapped_name = WT_CLASS_MAP.get(name)
            if mapped_name is None or mapped_name not in STANDARD_CLASSES:
                continue

            class_id = STANDARD_CLASSES[mapped_name]

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 转换为YOLO格式
            cx = (xmin + xmax) / 2 / img_width
            cy = (ymin + ymax) / 2 / img_height
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height

            # 确保在0-1范围内
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    except Exception as e:
        print(f"  [WARN] 解析XML失败 {xml_path}: {e}")

    return yolo_lines


def process_blade30_dataset(dataset_dir, dataset_name, output_dir):
    """处理Blade30数据集"""
    print(f"\n处理 {dataset_name}...")

    images = list(dataset_dir.rglob('*.jpg'))
    print(f"  找到 {len(images)} 张图片")

    processed = 0
    skipped = 0

    for img_path in images:
        # 查找对应JSON标注
        json_path = img_path.with_suffix('.json')

        if not json_path.exists():
            skipped += 1
            continue

        # 获取图片尺寸
        try:
            from PIL import Image
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception:
            img_width, img_height = 1920, 1080

        # 转换标注
        yolo_lines = convert_blade30_json_to_yolo(json_path, img_width, img_height)

        if not yolo_lines:
            skipped += 1
            continue

        # 复制图片
        new_name = f"{dataset_name}_{processed:06d}{img_path.suffix}"
        dst_img = output_dir / 'images' / new_name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_img)

        # 保存标注
        dst_label = output_dir / 'labels' / (dst_img.stem + '.txt')
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label, 'w') as f:
            f.write('\n'.join(yolo_lines))

        processed += 1

    print(f"  处理完成: {processed} 张, 跳过: {skipped} 张")
    return processed, skipped


def process_wt_dataset(dataset_dir, output_dir):
    """处理WT blade defect数据集"""
    print(f"\n处理 WT blade defect dataset...")

    # 读取数据划分
    split_file = dataset_dir / 'train_val_test_split.txt'
    splits = {}
    if split_file.exists():
        with open(split_file, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    splits[parts[0]] = parts[1]

    images = list((dataset_dir / 'JPEGImages').glob('*.jpg'))
    print(f"  找到 {len(images)} 张图片")

    processed = 0
    skipped = 0
    split_counts = Counter()

    for img_path in images:
        # 查找对应XML标注
        xml_path = dataset_dir / 'Annotations' / (img_path.stem + '.xml')

        if not xml_path.exists():
            skipped += 1
            continue

        # 转换标注
        yolo_lines = convert_voc_xml_to_yolo(xml_path)

        if not yolo_lines:
            skipped += 1
            continue

        # 获取数据划分
        split = splits.get(img_path.name, 'train')
        split_counts[split] += 1

        # 复制图片
        new_name = f"wt_{processed:06d}{img_path.suffix}"
        dst_img = output_dir / 'images' / split / new_name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_img)

        # 保存标注
        dst_label = output_dir / 'labels' / split / (dst_img.stem + '.txt')
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label, 'w') as f:
            f.write('\n'.join(yolo_lines))

        processed += 1

    print(f"  处理完成: {processed} 张, 跳过: {skipped} 张")
    print(f"  数据划分: {dict(split_counts)}")
    return processed, skipped, split_counts


def split_blade30_data(output_dir, ratios=(0.7, 0.15, 0.15), seed=42):
    """将Blade30数据划分到train/val/test"""
    print("\n划分Blade30数据...")

    random.seed(seed)

    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'

    # 获取所有图片
    images = list(images_dir.glob('*.jpg'))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        # 创建目录
        (output_dir / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split_name).mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            # 移动图片
            dst_img = output_dir / 'images' / split_name / img_path.name
            if img_path != dst_img:
                shutil.move(str(img_path), str(dst_img))

            # 移动标注
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                dst_label = output_dir / 'labels' / split_name / label_path.name
                shutil.move(str(label_path), str(dst_label))

        print(f"  {split_name}: {len(split_images)} 张")

    # 清理空目录
    for d in [images_dir, labels_dir]:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()


def merge_datasets():
    """合并所有数据集到最终目录"""
    print("\n合并数据集...")

    final_dir = PROCESSED_DIR / 'detection'
    final_dir.mkdir(parents=True, exist_ok=True)

    # 复制WT数据集 (已经划分好)
    wt_src = PROCESSED_DIR / 'wt_blade_defect'
    if wt_src.exists():
        for split in ['train', 'val', 'test']:
            src_img = wt_src / 'images' / split
            src_lbl = wt_src / 'labels' / split
            dst_img = final_dir / 'images' / split
            dst_lbl = final_dir / 'labels' / split

            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            if src_img.exists():
                for f in src_img.glob('*'):
                    shutil.copy2(str(f), str(dst_img / f.name))

            if src_lbl.exists():
                for f in src_lbl.glob('*'):
                    shutil.copy2(str(f), str(dst_lbl / f.name))

        print("  WT数据集已复制")

    # 复制Blade30数据集 (已经划分好)
    blade30_src = PROCESSED_DIR / 'blade30'
    if blade30_src.exists():
        for split in ['train', 'val', 'test']:
            src_img = blade30_src / 'images' / split
            src_lbl = blade30_src / 'labels' / split
            dst_img = final_dir / 'images' / split
            dst_lbl = final_dir / 'labels' / split

            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            if src_img.exists():
                for f in src_img.glob('*'):
                    shutil.copy2(str(f), str(dst_img / f.name))

            if src_lbl.exists():
                for f in src_lbl.glob('*'):
                    shutil.copy2(str(f), str(dst_lbl / f.name))

        print("  Blade30数据集已复制")

    # 统计最终数据集
    stats = {}
    for split in ['train', 'val', 'test']:
        img_dir = final_dir / 'images' / split
        lbl_dir = final_dir / 'labels' / split

        n_images = len(list(img_dir.glob('*'))) if img_dir.exists() else 0
        n_labels = len(list(lbl_dir.glob('*'))) if lbl_dir.exists() else 0

        stats[split] = {'images': n_images, 'labels': n_labels}
        print(f"  {split}: {n_images} 图片, {n_labels} 标注")

    return stats


def generate_eda_report():
    """生成EDA报告"""
    print("\n生成EDA报告...")

    final_dir = PROCESSED_DIR / 'detection'

    # 统计类别分布
    class_dist = Counter()
    ann_per_image = []
    image_sizes = []

    for split in ['train', 'val', 'test']:
        lbl_dir = final_dir / 'labels' / split
        if not lbl_dir.exists():
            continue

        for lbl_path in lbl_dir.glob('*.txt'):
            try:
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                ann_per_image.append(len(lines))
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        cname = [k for k, v in STANDARD_CLASSES.items() if v == class_id][0]
                        class_dist[cname] += 1
            except Exception:
                pass

    # 生成报告
    report_path = DOCS_DIR / 'eda_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# EDA Report - Wind Turbine Blade Defect Dataset\n\n")
        f.write("## Class Distribution\n\n")
        f.write("| Class | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        total = sum(class_dist.values()) or 1
        for cls, count in sorted(class_dist.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            f.write(f"| {cls} | {count} | {pct:.1f}% |\n")

        f.write("\n## Annotations per Image\n\n")
        if ann_per_image:
            f.write(f"- Min: {min(ann_per_image)}\n")
            f.write(f"- Max: {max(ann_per_image)}\n")
            f.write(f"- Avg: {sum(ann_per_image)/len(ann_per_image):.1f}\n")

        f.write("\n## Dataset Splits\n\n")
        f.write("| Split | Images |\n")
        f.write("|-------|--------|\n")
        for split in ['train', 'val', 'test']:
            img_dir = final_dir / 'images' / split
            n = len(list(img_dir.glob('*'))) if img_dir.exists() else 0
            f.write(f"| {split} | {n} |\n")

    print(f"  EDA报告已保存: {report_path}")


def main():
    print("=" * 60)
    print("风电叶片缺陷数据集完整处理流程")
    print("=" * 60)

    # Step 1: 处理Blade30 叶片1-15
    blade30_15_dir = BASE_DIR / '3_blade_1_15_with_labeldata' / '3_blade_1_15_with_labeldata'
    blade30_output = PROCESSED_DIR / 'blade30'
    blade30_output.mkdir(parents=True, exist_ok=True)

    if blade30_15_dir.exists():
        process_blade30_dataset(blade30_15_dir, 'blade30_1_15', blade30_output)

    # Step 2: 处理Blade30 叶片16-30
    blade30_30_dir = BASE_DIR / '3_blade_16_30_with_labeldata' / '3_blade_16_30_with_labeldata'
    if blade30_30_dir.exists():
        process_blade30_dataset(blade30_30_dir, 'blade30_16_30', blade30_output)

    # Step 3: 划分Blade30数据
    split_blade30_data(blade30_output)

    # Step 4: 处理WT数据集
    wt_dir = BASE_DIR / 'WT blade defect dataset' / 'WT blade defect dataset'
    wt_output = PROCESSED_DIR / 'wt_blade_defect'

    if wt_dir.exists():
        process_wt_dataset(wt_dir, wt_output)

    # Step 5: 合并数据集
    final_stats = merge_datasets()

    # Step 6: 生成EDA报告
    generate_eda_report()

    # 保存处理统计
    stats_path = DOCS_DIR / 'processing_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"最终数据集: {PROCESSED_DIR / 'detection'}")
    print(f"统计文件: {stats_path}")


if __name__ == '__main__':
    main()
