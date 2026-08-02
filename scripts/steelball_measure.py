"""
钢珠检测 + 直径测量（独立分支应用）

流程：
  1. YOLO 检测钢珠 → bbox
  2. 计算像素直径（bbox 宽高 / 圆拟合）
  3. 像素 → 物理直径（标定系数）

Usage:
    python scripts/steelball_measure.py --image img.jpg --model best.pt
    python scripts/steelball_measure.py --image img.jpg --real-diameter 1.0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_steelballs(model, image):
    """YOLO 检测钢珠，返回 bbox 列表 [x1,y1,x2,y2,conf]."""
    results = model(image)
    boxes = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                boxes.append([*xyxy, conf])
    return boxes


def pixel_diameter_from_bbox(box):
    """从 bbox 计算钢珠像素直径（钢珠近似圆形）. """
    x1, y1, x2, y2, _ = box
    w = x2 - x1
    h = y2 - y1
    # 钢珠是圆形，直径 ≈ bbox 平均边
    return (w + h) / 2.0


def pixel_to_physical(pixel_d, calib_k=None, real_d=None):
    """像素直径 → 物理直径.

    calib_k: 标定系数 (物理尺寸/像素)，已知则直接换算
    real_d: 若提供真实直径 + 像素直径，算出标定系数
    """
    if calib_k is not None:
        return pixel_d * calib_k
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--real-diameter", type=float, default=None,
                        help="钢珠真实直径 (cm)，用于标定/验证")
    parser.add_argument("--calib-k", type=float, default=None,
                        help="标定系数 (物理cm/像素)")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    print(f"检测: {args.image}")
    boxes = detect_steelballs(model, args.image)
    print(f"检测到 {len(boxes)} 个钢珠")

    if args.real_diameter and len(boxes) > 0:
        # 用第一个球标定（假设单球或同直径）
        pd = pixel_diameter_from_bbox(boxes[0])
        k = args.real_diameter / pd
        print(f"标定: 真实直径 {args.real_diameter}cm / 像素 {pd:.1f}px "
              f"→ 标定系数 {k:.5f} cm/px")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box
        pd = pixel_diameter_from_bbox(box)
        print(f"  球{i}: 中心=({(x1+x2)/2:.1f},{(y1+y2)/2:.1f}), "
              f"像素直径={pd:.1f}px, conf={conf:.2f}")
        if args.calib_k:
            ph = pixel_to_physical(pd, calib_k=args.calib_k)
            print(f"        物理直径 ≈ {ph:.3f} cm")


if __name__ == "__main__":
    main()
