"""
EDA探索性数据分析脚本
==================
功能：
1. 类别分布统计（柱状图）
2. 图片尺寸分布（直方图）
3. 标注框大小分布
4. 每张图片缺陷数量分布
5. 数据质量评估
6. 生成Markdown报告

使用方法：
    python eda_report.py --input processed/detection --output docs
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# 标准5类
STANDARD_CLASSES = {
    'crack': 0,
    'erosion': 1,
    'lightning': 2,
    'peeling': 3,
    'hole': 4,
}

# 中文类名
CLASS_NAMES_CN = {
    'crack': '裂纹',
    'erosion': '侵蚀',
    'lightning': '雷击',
    'peeling': '涂层脱落',
    'hole': '孔洞',
}


def find_split_dirs(base_dir):
    """查找train/val/test子目录"""
    base_dir = Path(base_dir)
    splits = {}
    for split in ['train', 'val', 'test']:
        img_dir = base_dir / 'images' / split
        lbl_dir = base_dir / 'labels' / split
        if img_dir.exists():
            splits[split] = (img_dir, lbl_dir if lbl_dir.exists() else None)
    return splits


def analyze_dataset(base_dir):
    """分析数据集"""
    splits = find_split_dirs(base_dir)

    stats = {
        'splits': {},
        'overall': {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': Counter(),
            'images_per_class': Counter(),
            'annotations_per_image': [],
            'image_sizes': [],
            'bbox_areas': [],
            'bbox_aspect_ratios': [],
        },
        'quality': {
            'empty_labels': 0,
            'invalid_annotations': 0,
            'tiny_boxes': 0,
            'large_boxes': 0,
        },
    }

    for split_name, (img_dir, lbl_dir) in splits.items():
        split_stats = {
            'images': 0,
            'annotations': 0,
            'class_distribution': Counter(),
        }

        images = list(img_dir.glob('*'))
        split_stats['images'] = len(images)
        stats['overall']['total_images'] += len(images)

        for img_path in images:
            if lbl_dir:
                label_path = lbl_dir / (img_path.stem + '.txt')
            else:
                label_path = None

            # 获取图片尺寸
            if HAS_PIL and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    stats['overall']['image_sizes'].append((w, h))
                except Exception:
                    pass

            ann_count = 0
            if label_path and label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                stats['quality']['invalid_annotations'] += 1
                                continue

                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])

                            # 类别统计
                            cname = [k for k, v in STANDARD_CLASSES.items() if v == class_id][0]
                            split_stats['class_distribution'][cname] += 1
                            stats['overall']['class_distribution'][cname] += 1
                            ann_count += 1

                            # 框面积和宽高比
                            area = w * h
                            stats['overall']['bbox_areas'].append(area)
                            if h > 0:
                                stats['overall']['bbox_aspect_ratios'].append(w / h)

                            # 质量检查
                            if area < 0.001:
                                stats['quality']['tiny_boxes'] += 1
                            if area > 0.5:
                                stats['quality']['large_boxes'] += 1

                            # 有效性检查
                            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                                stats['quality']['invalid_annotations'] += 1
                            if not (0 < w <= 1 and 0 < h <= 1):
                                stats['quality']['invalid_annotations'] += 1
                except Exception:
                    pass

            stats['overall']['annotations_per_image'].append(ann_count)
            stats['overall']['total_annotations'] += ann_count
            split_stats['annotations'] += ann_count

            if ann_count == 0:
                stats['quality']['empty_labels'] += 1

        stats['splits'][split_name] = {
            'images': split_stats['images'],
            'annotations': split_stats['annotations'],
            'class_distribution': dict(split_stats['class_distribution']),
        }

    # 转换Counter
    stats['overall']['class_distribution'] = dict(stats['overall']['class_distribution'])

    return stats


def generate_plots(stats, output_dir):
    """生成可视化图表"""
    if not HAS_MPL:
        print("[WARN] matplotlib 未安装，跳过图表生成")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    # 1. 类别分布柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(stats['overall']['class_distribution'].keys())
    counts = [stats['overall']['class_distribution'][c] for c in classes]
    cn_labels = [CLASS_NAMES_CN.get(c, c) for c in classes]
    colors = ['#2E86AB', '#28A745', '#FD7E14', '#6F42C1', '#DC3545']

    bars = ax.bar(cn_labels, counts, color=colors[:len(classes)], edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title('Category Distribution', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150)
    plt.close()

    # 2. 图片尺寸分布
    if stats['overall']['image_sizes']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        widths = [s[0] for s in stats['overall']['image_sizes']]
        heights = [s[1] for s in stats['overall']['image_sizes']]

        axes[0].hist(widths, bins=50, color='#2E86AB', edgecolor='white', alpha=0.8)
        axes[0].set_title('Image Width Distribution', fontsize=14)
        axes[0].set_xlabel('Width (px)')
        axes[0].set_ylabel('Count')

        axes[1].hist(heights, bins=50, color='#28A745', edgecolor='white', alpha=0.8)
        axes[1].set_title('Image Height Distribution', fontsize=14)
        axes[1].set_xlabel('Height (px)')
        axes[1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(output_dir / 'image_sizes.png', dpi=150)
        plt.close()

    # 3. 标注框面积分布
    if stats['overall']['bbox_areas']:
        fig, ax = plt.subplots(figsize=(10, 6))
        areas = [a * 100 for a in stats['overall']['bbox_areas']]  # 转为百分比
        ax.hist(areas, bins=50, color='#FD7E14', edgecolor='white', alpha=0.8)
        ax.set_title('Bounding Box Area Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Area (% of image)')
        ax.set_ylabel('Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / 'bbox_areas.png', dpi=150)
        plt.close()

    # 4. 每张图片缺陷数量分布
    if stats['overall']['annotations_per_image']:
        fig, ax = plt.subplots(figsize=(10, 6))
        ann_counts = stats['overall']['annotations_per_image']
        ax.hist(ann_counts, bins=range(0, max(ann_counts) + 2), color='#6F42C1',
                edgecolor='white', alpha=0.8)
        ax.set_title('Defects per Image Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Defects')
        ax.set_ylabel('Image Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / 'defects_per_image.png', dpi=150)
        plt.close()

    print(f"图表已保存到: {output_dir}")


def generate_report(stats, output_dir):
    """生成Markdown报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = stats['overall']
    quality = stats['quality']

    lines = []
    lines.append("# EDA Report - Wind Turbine Blade Defect Dataset\n")
    lines.append(f"Total Images: {overall['total_images']}")
    lines.append(f"Total Annotations: {overall['total_annotations']}\n")

    # 数据集划分
    lines.append("## Dataset Splits\n")
    lines.append("| Split | Images | Annotations |")
    lines.append("|-------|--------|-------------|")
    for split_name, split_stats in stats['splits'].items():
        lines.append(f"| {split_name} | {split_stats['images']} | {split_stats['annotations']} |")
    lines.append("")

    # 类别分布
    lines.append("## Class Distribution\n")
    lines.append("| Class | Chinese | Count | Percentage |")
    lines.append("|-------|---------|-------|------------|")
    total = sum(overall['class_distribution'].values()) or 1
    for cname, count in sorted(overall['class_distribution'].items(), key=lambda x: -x[1]):
        cn = CLASS_NAMES_CN.get(cname, cname)
        pct = count / total * 100
        lines.append(f"| {cname} | {cn} | {count} | {pct:.1f}% |")
    lines.append("")

    # 图片尺寸
    if overall['image_sizes']:
        widths = [s[0] for s in overall['image_sizes']]
        heights = [s[1] for s in overall['image_sizes']]
        lines.append("## Image Size Statistics\n")
        lines.append(f"- Width: min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.0f}")
        lines.append(f"- Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.0f}")
        lines.append("")

    # 标注框统计
    if overall['bbox_areas']:
        lines.append("## Bounding Box Statistics\n")
        areas = overall['bbox_areas']
        lines.append(f"- Area (%): min={min(areas)*100:.2f}, max={max(areas)*100:.2f}, avg={sum(areas)/len(areas)*100:.2f}")
        if overall['bbox_aspect_ratios']:
            ratios = overall['bbox_aspect_ratios']
            lines.append(f"- Aspect Ratio: min={min(ratios):.2f}, max={max(ratios):.2f}, avg={sum(ratios)/len(ratios):.2f}")
        lines.append("")

    # 每张图片缺陷数
    if overall['annotations_per_image']:
        anns = overall['annotations_per_image']
        lines.append("## Defects per Image\n")
        lines.append(f"- Min: {min(anns)}")
        lines.append(f"- Max: {max(anns)}")
        lines.append(f"- Avg: {sum(anns)/len(anns):.1f}")
        lines.append(f"- Median: {sorted(anns)[len(anns)//2]}")
        lines.append("")

    # 质量评估
    lines.append("## Quality Assessment\n")
    lines.append(f"- Empty labels (no annotations): {quality['empty_labels']}")
    lines.append(f"- Invalid annotations: {quality['invalid_annotations']}")
    lines.append(f"- Tiny boxes (area < 0.1%): {quality['tiny_boxes']}")
    lines.append(f"- Large boxes (area > 50%): {quality['large_boxes']}")
    lines.append("")

    # 图表引用
    lines.append("## Visualizations\n")
    lines.append("![Class Distribution](class_distribution.png)")
    lines.append("![Image Sizes](image_sizes.png)")
    lines.append("![BBox Areas](bbox_areas.png)")
    lines.append("![Defects per Image](defects_per_image.png)")

    report_path = output_dir / 'eda_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='EDA探索性数据分析')
    parser.add_argument('--input', required=True, help='数据目录')
    parser.add_argument('--output', default='docs', help='报告输出目录')
    args = parser.parse_args()

    print("=" * 60)
    print("风电叶片缺陷数据集 EDA 分析")
    print("=" * 60)

    stats = analyze_dataset(args.input)

    generate_plots(stats, args.output)
    generate_report(stats, args.output)

    # 保存原始数据
    raw_path = Path(args.output) / 'eda_stats.json'
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"原始数据已保存: {raw_path}")


if __name__ == '__main__':
    main()
