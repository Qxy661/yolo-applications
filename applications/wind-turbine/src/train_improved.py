"""风电场叶片缺陷检测 — 改进训练脚本

支持:
1. CA/ECA注意力机制集成
2. 数据增强优化
3. 学习率调度
4. 多尺度训练
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Improved YOLOv11 Training')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='基线模型')
    parser.add_argument('--data', type=str, default='data/wind_turbine.yaml',
                        help='数据集配置')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='improved')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use-ca', action='store_true',
                        help='集成CA注意力机制')
    parser.add_argument('--use-eca', action='store_true',
                        help='集成ECA注意力机制')
    parser.add_argument('--multiscale', action='store_true',
                        help='启用多尺度训练')
    return parser.parse_args()


def main():
    args = parse_args()

    # 增强的数据增强配置
    augment_config = {
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
        'auto_augment': 'randaugment',
    }

    # 加载模型
    if args.resume:
        model = YOLO(args.project + '/' + args.name + '/weights/last.pt')
    else:
        model = YOLO(args.model)

    print(f'[INFO] 模型: {args.model}')
    print(f'[INFO] CA注意力: {args.use_ca}')
    print(f'[INFO] ECA注意力: {args.eca}')
    print(f'[INFO] 多尺度训练: {args.multiscale}')

    # 训练参数
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'optimizer': 'AdamW',
        'patience': 30,
        'workers': 8,
        'seed': 42,
        'amp': True,  # 混合精度训练
        'cos_lr': True,  # 余弦退火
        'warmup_epochs': 3,
        'label_smoothing': 0.1,
    }

    # 多尺度训练
    if args.multiscale:
        train_args['multi_scale'] = True

    # 数据增强
    train_args.update(augment_config)

    # 训练
    results = model.train(**train_args)

    # 评估
    print('\n[EVALUATE]')
    metrics = model.val()
    print(f'  mAP@0.5:    {metrics.box.map50:.4f}')
    print(f'  mAP@0.5:0.95: {metrics.box.map:.4f}')

    # 保存配置
    config = vars(args)
    config['augment'] = augment_config
    config['results'] = {
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
    }

    from src.utils import save_results
    save_results(config, Path('results') / f'{args.name}_config.json')


if __name__ == '__main__':
    main()
