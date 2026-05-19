"""风电场叶片缺陷检测 — YOLOv11 推理脚本"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11 Wind Turbine Blade Defect Detection')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                        help='输入源 (图片/视频/目录)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='推理设备')
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存结果')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='结果保存目录')
    parser.add_argument('--name', type=str, default='inference',
                        help='实验名称')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='过滤类别ID')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载模型
    model = YOLO(args.weights)
    print(f'[INFO] 加载模型: {args.weights}')
    print(f'[INFO] 类别: {model.names}')

    # 推理
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        show=args.show,
        project=args.project,
        name=args.name,
        exist_ok=True,
        classes=args.classes,
    )

    # 统计结果
    total_detections = 0
    class_counts = {}
    for r in results:
        boxes = r.boxes
        total_detections += len(boxes)
        for cls in boxes.cls.cpu().numpy():
            cls_name = model.names[int(cls)]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    print(f'\n[RESULTS] 共检测到 {total_detections} 个缺陷')
    print('[CLASS COUNTS]')
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f'  {cls_name}: {count}')

    if args.save:
        print(f'\n[DONE] 结果保存至: {args.project}/{args.name}')


if __name__ == '__main__':
    main()
