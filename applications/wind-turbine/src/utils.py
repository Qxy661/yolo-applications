"""风电场叶片缺陷检测 — 工具函数"""
import os
import yaml
import json
from pathlib import Path
from datetime import datetime


def load_config(config_path):
    """加载YAML配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_results(results, output_path):
    """保存实验结果为JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results['timestamp'] = datetime.now().isoformat()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def count_dataset_classes(labels_dir, class_names):
    """统计数据集各类别数量"""
    counts = {name: 0 for name in class_names.values()}
    total_boxes = 0

    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in class_names:
                        counts[class_names[cls_id]] += 1
                    total_boxes += 1

    return counts, total_boxes


def print_dataset_stats(data_yaml_path):
    """打印数据集统计信息"""
    config = load_config(data_yaml_path)
    class_names = config['names']
    data_dir = Path(config['path'])

    print('=' * 60)
    print('数据集统计')
    print('=' * 60)

    for split in ['train', 'val', 'test']:
        labels_dir = data_dir / 'labels' / split
        if not labels_dir.exists():
            print(f'\n  {split}: 目录不存在')
            continue

        counts, total = count_dataset_classes(labels_dir, class_names)
        n_images = len(list((data_dir / 'images' / split).glob('*')))

        print(f'\n  {split} ({n_images} 张图片, {total} 个标注):')
        for name, count in counts.items():
            print(f'    {name:12s}: {count}')


class ExperimentTracker:
    """实验追踪器"""

    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []

    def log(self, name, metrics, config=None):
        """记录实验结果"""
        entry = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': config,
        }
        self.experiments.append(entry)

        # 保存到文件
        output = self.results_dir / f'{name}_metrics.json'
        save_results(entry, output)
        print(f'[LOG] 实验 {name} 结果已保存: {output}')

    def summary(self):
        """打印实验摘要"""
        print('\n' + '=' * 60)
        print('实验摘要')
        print('=' * 60)
        for exp in self.experiments:
            m = exp['metrics']
            print(f"  {exp['name']:20s}: mAP@0.5={m.get('mAP50', 0):.4f}, "
                  f"mAP@0.5:0.95={m.get('mAP50_95', 0):.4f}")
