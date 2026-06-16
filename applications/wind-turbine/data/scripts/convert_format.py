"""
格式转换脚本
============
功能：
1. JSON标注 → YOLO txt格式
2. VOC XML标注 → YOLO txt格式
3. 统一类别映射（5类标准）
4. 复制图片到标准目录

使用方法：
    python convert_format.py --input raw/blade30 --output processed/detection --format json
    python convert_format.py --input raw/qq6 --output processed/detection --format yolo
"""

import os
import sys
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# 标准5类映射
STANDARD_CLASSES = {
    'crack': 0,
    'erosion': 1,
    'lightning': 2,
    'peeling': 3,
    'hole': 4,
}

# 中文类别映射
CHINESE_CLASS_MAP = {
    '裂纹': 'crack',
    '侵蚀': 'erosion',
    '雷击': 'lightning',
    '涂层脱落': 'peeling',
    '涂层损伤': 'peeling',
    '剥落': 'peeling',
    '漆面': 'peeling',
    'paint': 'peeling',
    '孔洞': 'hole',
    'pin hole': 'hole',
    '损伤': 'erosion',
    'damaged': 'erosion',
}


def convert_json_to_yolo(json_path, img_width, img_height, class_mapping=None):
    """
    将JSON标注转换为YOLO格式
    JSON格式假设为:
    {
        "shapes": [
            {
                "label": "crack",
                "points": [[x1, y1], [x2, y2], ...]
            }
        ]
    }
    """
    if class_mapping is None:
        class_mapping = STANDARD_CLASSES

    yolo_lines = []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])

        for shape in shapes:
            label = shape.get('label', '').lower()

            # 映射类别
            mapped_label = class_mapping.get(label, None)
            if mapped_label is None:
                # 尝试中文映射
                mapped_label = CHINESE_CLASS_MAP.get(label, None)
            if mapped_label is None:
                print(f"  [WARN] 未知类别: {label}，跳过")
                continue

            if mapped_label not in STANDARD_CLASSES:
                print(f"  [WARN] 类别 {mapped_label} 不在标准5类中，跳过")
                continue

            class_id = STANDARD_CLASSES[mapped_label]

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
        print(f"  [ERROR] 解析JSON失败: {e}")

    return yolo_lines


def convert_xml_to_yolo(xml_path, class_mapping=None):
    """
    将VOC XML标注转换为YOLO格式
    """
    if class_mapping is None:
        class_mapping = STANDARD_CLASSES

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
            mapped_label = class_mapping.get(name, None)
            if mapped_label is None:
                mapped_label = CHINESE_CLASS_MAP.get(name, None)
            if mapped_label is None or mapped_label not in STANDARD_CLASSES:
                continue

            class_id = STANDARD_CLASSES[mapped_label]

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

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    except Exception as e:
        print(f"  [ERROR] 解析XML失败: {e}")

    return yolo_lines


def process_dataset(input_dir, output_dir, annotation_format='auto'):
    """
    处理整个数据集
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 创建输出目录
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)

    stats = {
        'total_images': 0,
        'converted': 0,
        'skipped': 0,
        'errors': 0,
        'class_distribution': {},
    }

    # 查找所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(input_dir.rglob(f'*{ext}'))
        images.extend(input_dir.rglob(f'*{ext.upper()}'))

    stats['total_images'] = len(images)
    print(f"\n找到 {len(images)} 张图片")

    # 处理每张图片
    for img_path in images:
        try:
            # 获取图片尺寸
            if HAS_PIL:
                img = Image.open(img_path)
                img_width, img_height = img.size
            else:
                # 默认尺寸
                img_width, img_height = 1920, 1080

            # 查找标注文件
            label_lines = []
            annotation_format_actual = annotation_format

            if annotation_format == 'auto':
                # 自动检测格式
                json_path = img_path.with_suffix('.json')
                xml_path = img_path.with_suffix('.xml')
                txt_path = img_path.with_suffix('.txt')

                if json_path.exists():
                    annotation_format_actual = 'json'
                    label_lines = convert_json_to_yolo(json_path, img_width, img_height)
                elif xml_path.exists():
                    annotation_format_actual = 'xml'
                    label_lines = convert_xml_to_yolo(xml_path)
                elif txt_path.exists():
                    annotation_format_actual = 'yolo'
                    # 已经是YOLO格式，直接复制
                    with open(txt_path, 'r') as f:
                        label_lines = [line.strip() for line in f if line.strip()]
                else:
                    stats['skipped'] += 1
                    continue
            elif annotation_format == 'json':
                json_path = img_path.with_suffix('.json')
                if json_path.exists():
                    label_lines = convert_json_to_yolo(json_path, img_width, img_height)
                else:
                    stats['skipped'] += 1
                    continue
            elif annotation_format == 'xml':
                xml_path = img_path.with_suffix('.xml')
                if xml_path.exists():
                    label_lines = convert_xml_to_yolo(xml_path)
                else:
                    stats['skipped'] += 1
                    continue
            elif annotation_format == 'yolo':
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists():
                    with open(txt_path, 'r') as f:
                        label_lines = [line.strip() for line in f if line.strip()]
                else:
                    stats['skipped'] += 1
                    continue

            # 复制图片
            dst_img = output_dir / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)

            # 保存标注
            dst_label = output_dir / 'labels' / (img_path.stem + '.txt')
            with open(dst_label, 'w') as f:
                f.write('\n'.join(label_lines))

            # 统计类别分布
            for line in label_lines:
                parts = line.split()
                if parts:
                    class_id = int(parts[0])
                    class_name = [k for k, v in STANDARD_CLASSES.items() if v == class_id][0]
                    stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1

            stats['converted'] += 1

        except Exception as e:
            print(f"  [ERROR] 处理 {img_path.name} 失败: {e}")
            stats['errors'] += 1

    # 输出统计
    print("\n" + "="*60)
    print("转换统计")
    print("="*60)
    print(f"总图片数: {stats['total_images']}")
    print(f"成功转换: {stats['converted']}")
    print(f"跳过: {stats['skipped']}")
    print(f"错误: {stats['errors']}")

    if stats['class_distribution']:
        print("\n类别分布:")
        for cls, count in sorted(stats['class_distribution'].items()):
            print(f"  {cls}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='标注格式转换')
    parser.add_argument('--input', required=True, help='输入目录')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--format', choices=['auto', 'json', 'xml', 'yolo'],
                        default='auto', help='标注格式')
    args = parser.parse_args()

    print("="*60)
    print("风电叶片缺陷数据集格式转换")
    print("="*60)

    stats = process_dataset(args.input, args.output, args.format)

    # 保存统计
    import json
    report_path = Path(args.output) / 'convert_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n统计报告已保存: {report_path}")


if __name__ == '__main__':
    HAS_PIL = True
    main()
