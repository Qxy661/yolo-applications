"""
Step 1: 数据整理脚本
==================
功能：
1. 扫描 raw/ 目录，统计各数据集信息
2. 生成数据清单 (inventory.json)
3. 验证图片和标注文件的对应关系
4. 输出统计报告

使用方法：
    python 01_organize_data.py
"""

import os
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# 配置
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'raw'
DOCS_DIR = BASE_DIR / 'docs'

# 标准5类
STANDARD_CLASSES = {
    'crack': 0,
    'erosion': 1,
    'lightning': 2,
    'peeling': 3,
    'hole': 4,
}


def calculate_file_hash(file_path):
    """计算文件MD5哈希"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def scan_blade30_dataset(dataset_dir, dataset_name):
    """扫描Blade30数据集"""
    stats = {
        'name': dataset_name,
        'path': str(dataset_dir),
        'type': 'blade30',
        'images': [],
        'annotations': [],
        'masks': [],
        'total_images': 0,
        'total_annotations': 0,
        'total_masks': 0,
        'class_distribution': Counter(),
    }

    # 查找所有图片
    for img_path in dataset_dir.rglob('*.jpg'):
        stats['images'].append({
            'path': str(img_path),
            'name': img_path.name,
            'size': img_path.stat().st_size,
        })
        stats['total_images'] += 1

    # 查找所有JSON标注
    for json_path in dataset_dir.rglob('*.json'):
        stats['annotations'].append({
            'path': str(json_path),
            'name': json_path.name,
            'format': 'labelme',
        })
        stats['total_annotations'] += 1

        # 解析类别
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                if label in STANDARD_CLASSES:
                    stats['class_distribution'][label] += 1
        except Exception:
            pass

    # 查找所有mask
    for mask_path in dataset_dir.rglob('mask/*.png'):
        stats['masks'].append({
            'path': str(mask_path),
            'name': mask_path.name,
        })
        stats['total_masks'] += 1

    stats['class_distribution'] = dict(stats['class_distribution'])
    return stats


def scan_wt_dataset(dataset_dir):
    """扫描WT blade defect数据集"""
    stats = {
        'name': 'WT blade defect dataset',
        'path': str(dataset_dir),
        'type': 'wt_blade_defect',
        'images': [],
        'annotations': [],
        'total_images': 0,
        'total_annotations': 0,
        'class_distribution': Counter(),
        'splits': {'train': 0, 'val': 0, 'test': 0},
    }

    # 读取类别定义
    class_defs = {}
    class_def_file = dataset_dir / 'class_definitions.txt'
    if class_def_file.exists():
        with open(class_def_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    class_defs[class_name] = class_id

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

    # 查找所有图片
    img_dir = dataset_dir / 'JPEGImages'
    if img_dir.exists():
        for img_path in img_dir.glob('*.jpg'):
            stats['images'].append({
                'path': str(img_path),
                'name': img_path.name,
                'size': img_path.stat().st_size,
                'split': splits.get(img_path.name, 'unknown'),
            })
            stats['total_images'] += 1

            # 统计划分
            split = splits.get(img_path.name, 'unknown')
            if split in stats['splits']:
                stats['splits'][split] += 1

    # 查找所有VOC XML标注
    ann_dir = dataset_dir / 'Annotations'
    if ann_dir.exists():
        for xml_path in ann_dir.glob('*.xml'):
            stats['annotations'].append({
                'path': str(xml_path),
                'name': xml_path.name,
                'format': 'voc_xml',
            })
            stats['total_annotations'] += 1

            # 解析类别
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    # 映射到标准类别
                    mapped = map_wt_class(name)
                    if mapped:
                        stats['class_distribution'][mapped] += 1
            except Exception:
                pass

    stats['class_distribution'] = dict(stats['class_distribution'])
    return stats


def map_wt_class(class_name):
    """映射WT数据集类别到标准类别"""
    mapping = {
        'craze': 'crack',
        'crack': 'crack',
        'hide_craze': 'crack',
        'corrosion': 'erosion',
        'surface_injure': 'erosion',
        'thunderstrike': 'lightning',
    }
    return mapping.get(class_name.lower())


def generate_inventory():
    """生成数据清单"""
    print("=" * 60)
    print("风电叶片缺陷数据集盘点")
    print("=" * 60)
    print(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据目录: {RAW_DIR}")
    print()

    inventory = {
        'scan_time': datetime.now().isoformat(),
        'base_dir': str(BASE_DIR),
        'raw_dir': str(RAW_DIR),
        'datasets': [],
        'summary': {
            'total_datasets': 0,
            'total_images': 0,
            'total_annotations': 0,
            'total_masks': 0,
            'class_distribution': Counter(),
        },
    }

    # 扫描Blade30 叶片1-15
    blade30_15_dir = BASE_DIR / '3_blade_1_15_with_labeldata' / '3_blade_1_15_with_labeldata'
    if blade30_15_dir.exists():
        print("\n[1/3] 扫描 Blade30 叶片1-15...")
        stats = scan_blade30_dataset(blade30_15_dir, 'blade30_1_15')
        inventory['datasets'].append(stats)
        print(f"  图片: {stats['total_images']}")
        print(f"  标注: {stats['total_annotations']}")
        print(f"  掩码: {stats['total_masks']}")
        print(f"  类别: {stats['class_distribution']}")

    # 扫描Blade30 叶片16-30
    blade30_30_dir = BASE_DIR / '3_blade_16_30_with_labeldata' / '3_blade_16_30_with_labeldata'
    if blade30_30_dir.exists():
        print("\n[2/3] 扫描 Blade30 叶片16-30...")
        stats = scan_blade30_dataset(blade30_30_dir, 'blade30_16_30')
        inventory['datasets'].append(stats)
        print(f"  图片: {stats['total_images']}")
        print(f"  标注: {stats['total_annotations']}")
        print(f"  掩码: {stats['total_masks']}")
        print(f"  类别: {stats['class_distribution']}")

    # 扫描WT blade defect dataset
    wt_dir = BASE_DIR / 'WT blade defect dataset' / 'WT blade defect dataset'
    if wt_dir.exists():
        print("\n[3/3] 扫描 WT blade defect dataset...")
        stats = scan_wt_dataset(wt_dir)
        inventory['datasets'].append(stats)
        print(f"  图片: {stats['total_images']}")
        print(f"  标注: {stats['total_annotations']}")
        print(f"  类别: {stats['class_distribution']}")
        print(f"  划分: {stats['splits']}")

    # 汇总统计
    for dataset in inventory['datasets']:
        inventory['summary']['total_datasets'] += 1
        inventory['summary']['total_images'] += dataset['total_images']
        inventory['summary']['total_annotations'] += dataset.get('total_annotations', 0)
        inventory['summary']['total_masks'] += dataset.get('total_masks', 0)
        for cls, count in dataset.get('class_distribution', {}).items():
            inventory['summary']['class_distribution'][cls] += count

    inventory['summary']['class_distribution'] = dict(inventory['summary']['class_distribution'])

    # 输出汇总
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    print(f"数据集数量: {inventory['summary']['total_datasets']}")
    print(f"总图片数: {inventory['summary']['total_images']}")
    print(f"总标注数: {inventory['summary']['total_annotations']}")
    print(f"总掩码数: {inventory['summary']['total_masks']}")
    print(f"\n类别分布:")
    for cls, count in sorted(inventory['summary']['class_distribution'].items()):
        print(f"  {cls}: {count}")

    return inventory


def save_inventory(inventory):
    """保存数据清单"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 保存JSON清单
    json_path = DOCS_DIR / 'inventory.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n数据清单已保存: {json_path}")

    # 保存可读的文本报告
    report_path = DOCS_DIR / 'data_inventory_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 数据集盘点报告\n\n")
        f.write(f"> 生成时间: {inventory['scan_time']}\n\n")
        f.write("## 汇总统计\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 数据集数量 | {inventory['summary']['total_datasets']} |\n")
        f.write(f"| 总图片数 | {inventory['summary']['total_images']} |\n")
        f.write(f"| 总标注数 | {inventory['summary']['total_annotations']} |\n")
        f.write(f"| 总掩码数 | {inventory['summary']['total_masks']} |\n\n")

        f.write("## 类别分布\n\n")
        f.write("| 类别 | 数量 | 占比 |\n")
        f.write("|------|------|------|\n")
        total = sum(inventory['summary']['class_distribution'].values()) or 1
        for cls, count in sorted(inventory['summary']['class_distribution'].items()):
            pct = count / total * 100
            f.write(f"| {cls} | {count} | {pct:.1f}% |\n")

        f.write("\n## 各数据集详情\n\n")
        for dataset in inventory['datasets']:
            f.write(f"### {dataset['name']}\n\n")
            f.write(f"- 路径: `{dataset['path']}`\n")
            f.write(f"- 类型: {dataset['type']}\n")
            f.write(f"- 图片数: {dataset['total_images']}\n")
            f.write(f"- 标注数: {dataset.get('total_annotations', 0)}\n")
            if dataset.get('total_masks'):
                f.write(f"- 掩码数: {dataset['total_masks']}\n")
            if dataset.get('class_distribution'):
                f.write(f"- 类别分布: {dataset['class_distribution']}\n")
            if dataset.get('splits'):
                f.write(f"- 数据划分: {dataset['splits']}\n")
            f.write("\n")

    print(f"文本报告已保存: {report_path}")


def main():
    inventory = generate_inventory()
    save_inventory(inventory)
    print("\n数据盘点完成！")


if __name__ == '__main__':
    main()
