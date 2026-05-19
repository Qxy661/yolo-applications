"""风电场叶片缺陷检测 — YOLOv11 评估脚本"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11 Model Evaluation')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data', type=str, default='data/wind_turbine.yaml',
                        help='数据集配置')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入尺寸')
    parser.add_argument('--batch', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--device', type=str, default='0',
                        help='设备')
    parser.add_argument('--save_json', action='store_true',
                        help='保存JSON结果')
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)
    print(f'[INFO] 评估模型: {args.weights}')

    # 评估
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split='val',
    )

    # 打印详细结果
    print('\n' + '=' * 60)
    print('评估结果')
    print('=' * 60)
    print(f'  mAP@0.5:       {metrics.box.map50:.4f}')
    print(f'  mAP@0.5:0.95:  {metrics.box.map:.4f}')
    print(f'  Precision:     {metrics.box.mp:.4f}')
    print(f'  Recall:        {metrics.box.mr:.4f}')
    print(f'  F1-Score:      {2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8):.4f}')

    # 各类别结果
    print('\n[PER-CLASS RESULTS]')
    class_names = model.names
    for i, (map50, map50_95) in enumerate(zip(metrics.box.maps50, metrics.box.maps)):
        print(f'  {class_names[i]:12s}: mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}')

    # 保存结果
    if args.save_json:
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': args.weights,
            'metrics': {
                'mAP50': float(metrics.box.map50),
                'mAP50_95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
            },
            'per_class': {
                class_names[i]: {
                    'mAP50': float(metrics.box.maps50[i]),
                    'mAP50_95': float(metrics.box.maps[i]),
                }
                for i in range(len(class_names))
            },
        }
        output_path = Path('results') / f'eval_{Path(args.weights).stem}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'\n[SAVE] 结果已保存: {output_path}')


if __name__ == '__main__':
    main()
