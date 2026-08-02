"""
钢珠检测 + 直径测量 + 高光校验（独立分支应用）

主流程（通用闭环）：YOLO 检测 → bbox → 直径测量
辅助手段（钢珠特定 trick）：高光点校验，可开关，不污染通用框架

Usage:
    python scripts/steelball_measure.py --image img.jpg --model best.pt
    python scripts/steelball_measure.py --image img.jpg --real-diameter 1.0 --use-highlight
    python scripts/steelball_measure.py --image img.jpg --visualize --out result.jpg
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
    """在 bbox 内找强反光点（高光），返回 (存在, 偏移, 质心)."""
    x1, y1, x2, y2, _ = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return False, 1.0, (0, 0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    highlight_mask = gray > threshold
    if not highlight_mask.any():
        return False, 1.0, (0, 0)

    ys, xs = np.where(highlight_mask)
    cx = xs.mean()
    cy = ys.mean()
    off_x = (cx - (x2 - x1) / 2) / (x2 - x1)
    off_y = (cy - (y2 - y1) / 2) / (y2 - y1)
    offset = np.sqrt(off_x ** 2 + off_y ** 2)
    # 全局坐标的高光质心
    gx, gy = x1 + cx, y1 + cy
    return True, offset, (gx, gy)


def verify_with_highlight(img, box, max_offset=0.3, threshold=200):
    """高光点校验 bbox（双保险）."""
    has_highlight, offset, centroid = find_specular_highlight(img, box, threshold)
    confirmed = has_highlight and offset < max_offset
    return confirmed, offset, centroid


def pixel_diameter_from_bbox(box):
    """像素直径（钢珠近似圆形）."""
    x1, y1, x2, y2, _ = box
    return (x2 - x1 + y2 - y1) / 2.0


def pixel_to_physical(pixel_d, calib_k=None):
    if calib_k is not None:
        return pixel_d * calib_k
    return None


def visualize(img, boxes, highlight_results, out_path, calib_k=None):
    """可视化检测结果：bbox + 高光点 + 确认状态."""
    vis = img.copy()
    for i, (box, (confirmed, offset, centroid)) in enumerate(
            zip(boxes, highlight_results)):
        x1, y1, x2, y2, conf = [int(v) for v in box]
        # 颜色：确认=绿，疑似误检=红
        color = (0, 255, 0) if confirmed else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        # 高光点
        gx, gy = int(centroid[0]), int(centroid[1])
        cv2.circle(vis, (gx, gy), 3, (0, 165, 255), -1)  # 橙色高光点

        pd = pixel_diameter_from_bbox(box)
        ph = pixel_to_physical(pd, calib_k)
        label = f"d={ph:.2f}cm" if ph else f"{pd:.0f}px"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(out_path, vis)
    print(f"可视化已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--real-diameter", type=float, default=None)
    parser.add_argument("--calib-k", type=float, default=None)
    parser.add_argument("--use-highlight", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--out", default="result.jpg")
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
        print(f"标定: {args.real_diameter}cm / {pd:.1f}px → 系数 {k:.5f} cm/px")
        args.calib_k = args.calib_k or k

    highlight_results = []
    kept = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box
        pd = pixel_diameter_from_bbox(box)

        note = ""
        confirmed = None
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        if args.use_highlight:
            confirmed, offset, centroid = verify_with_highlight(img, box)
            if confirmed:
                note = f" [高光✓ offset={offset:.2f}]"
                kept += 1
            else:
                note = f" [高光✗ offset={offset:.2f}]"
        highlight_results.append((confirmed if confirmed is not None else True,
                                  offset if args.use_highlight else 0,
                                  centroid))

        print(f"  球{i}: 中心=({(x1+x2)/2:.1f},{(y1+y2)/2:.1f}), "
              f"像素直径={pd:.1f}px, conf={conf:.2f}{note}")
        if args.calib_k:
            ph = pixel_to_physical(pd, calib_k=args.calib_k)
            print(f"        物理直径 ≈ {ph:.3f} cm")

    if args.use_highlight:
        print(f"\n高光校验: {kept}/{len(boxes)} 被确认")

    if args.visualize:
        visualize(img, boxes, highlight_results, args.out, args.calib_k)


if __name__ == "__main__":
    main()
