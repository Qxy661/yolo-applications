"""风电场叶片缺陷检测 — YOLOv11 训练脚本"""
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11 Wind Turbine Blade Defect Training')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='模型配置 (yolo11n.pt/yolo11s.pt/yolo11m.pt)')
    parser.add_argument('--data', type=str, default='data/wind_turbine.yaml',
                        help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮次')
    parser.add_argument('--batch', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备 (0/cpu)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='结果保存目录')
    parser.add_argument('--name', type=str, default='baseline',
                        help='实验名称')
    parser.add_argument('--resume', action='store_true',
                        help='恢复训练')
    parser.add_argument('--pretrained', type=str, default='yolo11n.pt',
                        help='预训练权重路径')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='优化器 (SGD/Adam/AdamW)')
    parser.add_argument('--patience', type=int, default=30,
                        help='早停轮次')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def main():
    args = parse_args()

    # 数据增强配置
    default_augment = {
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.9,
        'fliplr': 0.5,
        'flipud': 0.0,
        'erasing': 0.4,
    }

    # 加载模型
    if args.resume:
        model = YOLO(args.project + '/' + args.name + '/weights/last.pt')
        print(f'[RESUME] 从 {model.model_name} 恢复训练')
    else:
        model = YOLO(args.model)
        print(f'[INFO] 加载模型: {args.model}')

    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=args.pretrained if not args.resume else True,
        optimizer=args.optimizer,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        augment=True,
        **default_augment,
    )

    # 评估
    print('\n[EVALUATE] 在验证集上评估...')
    metrics = model.val()
    print(f'\n[RESULTS]')
    print(f'  mAP@0.5:    {metrics.box.map50:.4f}')
    print(f'  mAP@0.5:0.95: {metrics.box.map:.4f}')
    print(f'  Precision:  {metrics.box.mp:.4f}')
    print(f'  Recall:     {metrics.box.mr:.4f}')

    # 保存最佳模型路径
    best_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    print(f'\n[DONE] 最佳模型: {best_path}')


if __name__ == '__main__':
    main()
