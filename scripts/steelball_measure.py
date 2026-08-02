"""
钢珠检测 + 直径测量（独立分支应用）
含"反光作为特征"的辅助校验手段（trick，非通用框架）。

主流程（通用闭环）：YOLO 检测 → bbox → 直径测量
辅助手段（钢珠特定）：高光点校验 —— 反光是钢珠的稳定特征，
  用于确认 bbox 真实性、提高检测可靠性。

Usage:
    python scripts/steelball_measure.py --image img.jpg --model best.pt
    python scripts/steelball_measure.py --image img.jpg --real-diameter 1.0
"""
import argparse
import os
import sys

import cv2
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


def find_specular_highlight(img, box, threshold=200):
    """在 bbox 内找强反光点（高光）。

    钢珠是金属球，镜面反射产生集中高光 → 高光点是钢珠的稳定特征。
    返回: (存在, 高光点相对 bbox 中心的偏移比例)。
    """
    x1, y1, x2, y2, _ = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return False, 1.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 高光点 = 灰度超过阈值的区域
    highlight_mask = gray > threshold
    if not highlight_mask.any():
        return False, 1.0

    # 高光点质心（应在球心附近）
    ys, xs = np.where(highlight_mask)
    cx = xs.mean()
    cy = ys.mean()

    # 相对 bbox 中心的偏移（归一化，中心=0，边缘=0.5）
    off_x = (cx - (x2 - x1) / 2) / (x2 - x1)
    off_y = (cy - (y2 - y1) / 2) / (y2 - y1)
    offset = np.sqrt(off_x ** 2 + off_y ** 2)
    return True, offset


def verify_with_highlight(img, box, max_offset=0.3, threshold=200):
    """用高光点校验 bbox 真实性（双保险）。

    反光是钢珠的稳定特征：若 bbox 内有高光点且接近中心 → 确认钢珠。
    返回: (是否确认, 高光点偏移)。
    """
    has_highlight, offset = find_specular_highlight(img, box, threshold)
    confirmed = has_highlight and offset < max_offset
    return confirmed, offset


def pixel_diameter_from_bbox(box):
    """从 bbox 计算钢珠像素直径（钢珠近似圆形）. """
    x1, y1, x2, y2, _ = box
    w = x2 - x1
    h = y2 - y1
    return (w + h) / 2.0


def pixel_to_physical(pixel_d, calib_k=None, real_d=None):
    """像素直径 → 物理直径."""
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
    parser.add_argument("--use-highlight", action="store_true",
                        help="启用高光点校验（钢珠特定手段）")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    img = cv2.imread(args.image)
    if img is None:
        print(f"无法读取: {args.image}")
        return

    print(f"检测: {args.image}")
    boxes = detect_steelballs(model, img)
    print(f"检测到 {len(boxes)} 个钢珠")

    if args.real_diameter and len(boxes) > 0:
        pd = pixel_diameter_from_bbox(boxes[0])
        k = args.real_diameter / pd
        print(f"标定: 真实直径 {args.real_diameter}cm / 像素 {pd:.1f}px "
              f"→ 标定系数 {k:.5f} cm/px")

    kept = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box
        pd = pixel_diameter_from_bbox(box)

        # 高光校验（辅助手段，可开关）
        highlight_note = ""
        if args.use_highlight:
            confirmed, offset = verify_with_highlight(img, box)
            if confirmed:
                highlight_note = f" [高光✓ offset={offset:.2f}]"
                kept += 1
            else:
                highlight_note = f" [高光✗ 疑似误检 offset={offset:.2f}]"

        print(f"  球{i}: 中心=({(x1+x2)/2:.1f},{(y1+y2)/2:.1f}), "
              f"像素直径={pd:.1f}px, conf={conf:.2f}{highlight_note}")
        if args.calib_k:
            ph = pixel_to_physical(pd, calib_k=args.calib_k)
            print(f"        物理直径 ≈ {ph:.3f} cm")

    if args.use_highlight:
        print(f"\n高光校验: {kept}/{len(boxes)} 个被确认为钢珠")


if __name__ == "__main__":
    main()
