"""
钢珠测量全景图生成（论文级多图拼接）

对多张代表性钢珠图做 检测 + 高光校验 + 直径标注，拼成全景：
  - 每张图：绿色确认框 / 红色疑似框 + 橙色高光点 + 直径标注(px 或 cm)
  - 覆盖场景：密集多球 / 高光小目标 / 反光校验
用法：
  python scripts/steelball_measure_panorama.py \
      --model applications/steel-ball/results/best.pt \
      --images list.txt --real-diameter 1.0 --use-highlight \
      --out results/steelball/measure_panorama.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.steelball_measure import (
    detect_steelballs, find_specular_highlight, verify_with_highlight,
    pixel_diameter_from_bbox, pixel_to_physical,
)


def annotate(img, boxes, use_highlight, calib_k=None):
    """对一张图做测量标注：框 + 高光点 + 直径标签. 返回标注图."""
    vis = img.copy()
    n_confirm = 0
    for box in boxes:
        x1, y1, x2, y2, conf = [int(v) for v in box]
        confirmed = True
        if use_highlight:
            confirmed, offset, centroid = verify_with_highlight(img, box)
            color = (0, 255, 0) if confirmed else (0, 0, 255)
            gx, gy = int(centroid[0]), int(centroid[1])
            cv2.circle(vis, (gx, gy), 4, (0, 165, 255), -1)  # 橙色高光点
        else:
            color = (0, 255, 0)
        if confirmed:
            n_confirm += 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        pd = pixel_diameter_from_bbox(box)
        ph = pixel_to_physical(pd, calib_k)
        label = f"d={ph:.2f}cm" if ph else f"{pd:.0f}px"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis, n_confirm


def make_panorama(model, image_paths, use_highlight, calib_k, cols=3, cell=640):
    """拼多张测量图为全景（统一格子）."""
    drawn = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        boxes = detect_steelballs(model, img)
        vis, n = annotate(img, boxes, use_highlight, calib_k)
        # cover 统一格子
        h, w = vis.shape[:2]
        scale = max(cell / w, cell / h)
        nw, nh = int(w * scale), int(h * scale)
        vis = cv2.resize(vis, (nw, nh))
        x, y = (nw - cell) // 2, (nh - cell) // 2
        vis = vis[y:y + cell, x:x + cell]
        # 底部加说明条
        bar = np.ones((36, cell, 3), dtype=np.uint8) * 245
        tag = f"{len(boxes)} balls | {n}/{len(boxes)} highlight-confirmed"
        cv2.putText(bar, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (40, 40, 40), 1, cv2.LINE_AA)
        drawn.append(np.vstack([vis, bar]))

    rows = (len(drawn) + cols - 1) // cols
    canvas = np.ones(((cell + 36) * rows, cell * cols, 3), dtype=np.uint8) * 255
    for i, im in enumerate(drawn):
        r, c = i // cols, i % cols
        h, w = im.shape[:2]
        canvas[r * (cell + 36):r * (cell + 36) + h,
               c * cell:c * cell + w] = im
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True, help="图片列表txt或glob")
    parser.add_argument("--real-diameter", type=float, default=None, help="真实直径cm(标定)")
    parser.add_argument("--use-highlight", action="store_true")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--out", default="measure_panorama.jpg")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    if args.images.endswith(".txt"):
        paths = [l.strip() for l in open(args.images) if l.strip()]
    else:
        import glob
        paths = sorted(glob.glob(args.images))

    # 标定：用第一张图第一个球估算 cm/px
    calib_k = None
    if args.real_diameter:
        img0 = cv2.imread(paths[0])
        boxes = detect_steelballs(model, img0)
        if boxes:
            pd = pixel_diameter_from_bbox(boxes[0])
            calib_k = args.real_diameter / pd
            print(f"标定: {args.real_diameter}cm / {pd:.1f}px → {calib_k:.5f} cm/px")

    canvas = make_panorama(model, paths, args.use_highlight, calib_k, args.cols)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f"测量全景已保存: {args.out} ({len(paths)}张)")


if __name__ == "__main__":
    main()
