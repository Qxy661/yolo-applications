"""
风电叶片缺陷检测 - 模型对比实验
================================
对比不同YOLO版本和改进模型的性能:
1. YOLOv5n - 经典轻量级基线
2. YOLOv8n - 当前主流
3. YOLOv11n - 原始基线
4. YOLOv11n-Improved - 本方案改进版

使用方法：
    python experiments/compare_models.py
"""

import json
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = 'E:/yolo-wind-turbine/data/wind_turbine_2cls.yaml'
PROJECT = 'E:/yolo-wind-turbine/runs/detect'

# 对比模型 - 使用已训练的best.pt
MODELS = [
    ('yolov5n.pt', 'compare_yolov5n', 'YOLOv5n'),
    ('yolov8n.pt', 'compare_yolov8n', 'YOLOv8n'),
    ('yolo11n.pt', None, 'YOLOv11n (Baseline)'),
    (f'{PROJECT}/finetune_improved_v2/weights/best.pt', None, 'YOLOv11n-Improved'),
]


def eval_model(model_path, name, desc):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估模型: {desc}")
    print(f"{'='*60}")

    try:
        model = YOLO(model_path)
        metrics = model.val(data=DATA_YAML, split='val')

        result = {
            'model': model_path,
            'name': name,
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
        return {'model': model_path, 'name': name, 'desc': desc, 'error': str(e)}


def train_and_eval(model_path, name, desc):
    """训练并评估模型"""
    print(f"\n{'='*60}")
    print(f"训练+评估: {desc}")
    print(f"{'='*60}")

    try:
        model = YOLO(model_path)
        model.train(
            data=DATA_YAML, epochs=100, imgsz=640, batch=8, device=0, workers=4,
            mosaic=1.0, mixup=0.1, copy_paste=0.1, patience=20,
            optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937,
            weight_decay=0.0005, warmup_epochs=3, warmup_momentum=0.8,
            project=PROJECT, name=f'compare_{name}', exist_ok=True, verbose=True,
        )
        best = YOLO(f'{PROJECT}/compare_{name}/weights/best.pt')
        metrics = best.val(data=DATA_YAML, split='val')
        return {
            'model': model_path, 'name': name, 'desc': desc,
            'map50': round(metrics.box.map50, 4),
            'map50_95': round(metrics.box.map, 4),
            'precision': round(metrics.box.mp, 4),
            'recall': round(metrics.box.mr, 4),
        }
    except Exception as e:
        print(f"  失败: {e}")
        return {'model': model_path, 'name': name, 'desc': desc, 'error': str(e)}


def main():
    print("=" * 60)
    print("风电叶片缺陷检测 - 模型对比实验")
    print("=" * 60)

    results = []

    # v5n和v8n需要训练
    for model_path, name, desc in MODELS[:2]:
        r = train_and_eval(model_path, name, desc)
        results.append(r)

    # v11n baseline直接评估
    r = eval_model('yolo11n.pt', 'yolo11n', 'YOLOv11n (Baseline)')
    results.append(r)

    # improved使用已训练的finetune模型
    imp_path = f'{PROJECT}/finetune_improved_v2/weights/best.pt'
    if Path(imp_path).exists():
        r = eval_model(imp_path, 'improved', 'YOLOv11n-Improved')
    else:
        print(f"\n改进模型尚未训练: {imp_path}")
        r = {'desc': 'YOLOv11n-Improved', 'error': 'not trained'}
    results.append(r)

    # 汇总
    print("\n" + "=" * 70)
    print("模型对比结果汇总")
    print("=" * 70)
    print(f"\n{'模型':<25} {'mAP@0.5':<10} {'mAP@0.5:0.95':<15} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    for r in results:
        if 'error' not in r:
            print(f"{r['desc']:<25} {r['map50']:<10.4f} {r['map50_95']:<15.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}")
        else:
            print(f"{r['desc']:<25} {'FAILED':<10}")

    out = Path('E:/yolo-wind-turbine/data/docs/comparison_results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out}")


if __name__ == '__main__':
    main()
