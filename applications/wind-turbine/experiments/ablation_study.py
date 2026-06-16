"""
风电叶片缺陷检测 - 消融实验
================================
验证各改进模块的独立贡献:
1. Baseline - 原始YOLOv11n
2. +C2PSA增强 - 注意力机制
3. +Lightweight - 轻量化设计
4. +BiFPN - 特征融合(冻结backbone)
5. All - 综合改进

使用方法：
    python experiments/ablation_study.py
"""

import json
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = 'E:/yolo-wind-turbine/data/wind_turbine_2cls.yaml'
PROJECT = 'E:/yolo-wind-turbine/runs/detect'

# 消融实验 - 使用已训练的finetune模型
ABLATION_MODELS = [
    ('yolo11n.pt', None, 'Baseline (原始YOLOv11n)'),
    (f'{PROJECT}/finetune_ca_v2/weights/best.pt', None, '+C2PSA增强 (注意力机制)'),
    (f'{PROJECT}/finetune_light_v2/weights/best.pt', None, '+Lightweight (轻量化)'),
    (f'{PROJECT}/finetune_bifpn_v2/weights/best.pt', None, '+BiFPN (特征融合)'),
    (f'{PROJECT}/finetune_improved_v2/weights/best.pt', None, 'All (综合改进)'),
]


def eval_model(model_path, desc):
    """评估模型"""
    print(f"\n{'='*60}")
    print(f"消融实验: {desc}")
    print(f"{'='*60}")

    try:
        model = YOLO(model_path)
        metrics = model.val(data=DATA_YAML, split='val')

        result = {
            'model': model_path,
            'desc': desc,
            'map50': round(metrics.box.map50, 4),
            'map50_95': round(metrics.box.map, 4),
            'precision': round(metrics.box.mp, 4),
            'recall': round(metrics.box.mr, 4),
        }
        print(f"  mAP@0.5:     {result['map50']:.4f}")
        print(f"  mAP@0.5:0.95: {result['map50_95']:.4f}")
        print(f"  Precision:   {result['precision']:.4f}")
        print(f"  Recall:      {result['recall']:.4f}")
        return result

    except Exception as e:
        print(f"  评估失败: {e}")
        return {'model': model_path, 'desc': desc, 'error': str(e)}


def main():
    print("=" * 60)
    print("风电叶片缺陷检测 - 消融实验")
    print("=" * 60)

    results = []
    for model_path, name, desc in ABLATION_MODELS:
        r = eval_model(model_path, desc)
        results.append(r)

    # 汇总
    baseline_map50 = results[0].get('map50', 0) if 'error' not in results[0] else 0
    print("\n" + "=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)
    print(f"\n{'配置':<30} {'mAP@0.5':<10} {'变化':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    for r in results:
        if 'error' not in r:
            diff = ((r['map50'] - baseline_map50) / baseline_map50 * 100) if baseline_map50 else 0
            print(f"{r['desc']:<30} {r['map50']:<10.4f} {diff:+.1f}%{'':<5} {r['precision']:<10.4f} {r['recall']:<10.4f}")
        else:
            print(f"{r['desc']:<30} {'FAILED':<10}")

    out = Path('E:/yolo-wind-turbine/data/docs/ablation_results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out}")


if __name__ == '__main__':
    main()
