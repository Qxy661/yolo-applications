"""
检测结果全景图生成（论文级多图拼接）

用打磨风格（细框/小字体/无score）绘制多张检测图，拼成网格全景。
适用于：小目标 VisDrone / 钢珠检测 的多图展示。

Usage:
    python scripts/detection_panorama.py --images dir/*.jpg --model best.pt --out panorama.jpg
"""
import argparse
import glob
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO


def detect_clean(model, img, conf=0.25, iou_thresh=0.45):
    """显式 NMS 检测（解决重复框）."""
    results = model(img, conf=min(conf, 0.05), verbose=False)
    all_boxes = []
    for r in results:
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu()
            confs = r.boxes.conf.cpu()
            for i in range(len(xyxy)):
                all_boxes.append([*xyxy[i].tolist(), confs[i].item()])
    if not all_boxes:
        return []
    boxes_t = torch.tensor([b[:4] for b in all_boxes])
    scores_t = torch.tensor([b[4] for b in all_boxes])
    keep = nms(boxes_t, scores_t, iou_threshold=iou_thresh)
    return [all_boxes[i] for i in keep.tolist() if all_boxes[i][4] >= conf]


def draw_clean(img, boxes, label="object", color=(0, 255, 0)):
    """精细绘制：细框 + 小字体 + 不显示 score."""
    vis = img.copy()
    h, w = img.shape[:2]
    font_scale = 0.4 if min(h, w) > 400 else 0.3
    line_width = 1

    for b in boxes:
        x1, y1, x2, y2, c = [int(v) if i < 4 else v for i, v in enumerate(b)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_width)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.putText(vis, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    return vis


def make_panorama(model, images, label, color, cols=3, max_w=480):
    """绘制多张检测图并拼成网格全景."""
    drawn = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        # 缩放
        h, w = img.shape[:2]
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        boxes = detect_clean(model, img)
        vis = draw_clean(img, boxes, label, color)
        drawn.append(vis)

    # 拼成网格
    rows = (len(drawn) + cols - 1) // cols
    cell_h = max(h.shape[0] for h in drawn)
    cell_w = max(h.shape[1] for h in drawn)
    canvas = np.ones((cell_h * rows, cell_w * cols, 3), dtype=np.uint8) * 255
    for i, img in enumerate(drawn):
        r, c = i // cols, i % cols
        h, w = img.shape[:2]
        canvas[r * cell_h:r * cell_h + h, c * cell_w:c * cell_w + w] = img
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="图片路径或glob")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="object")
    parser.add_argument("--color", default="green", choices=["green", "orange", "blue"])
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--out", default="panorama.jpg")
    args = parser.parse_args()

    color = {"green": (0, 255, 0), "orange": (0, 165, 255), "blue": (255, 0, 0)}[args.color]
    images = sorted(glob.glob(args.images))
    model = YOLO(args.model)
    canvas = make_panorama(model, images[:9], args.label, color, args.cols)
    cv2.imwrite(args.out, canvas)
    print(f"全景图已保存: {args.out}, 包含 {min(len(images), 9)} 张")


if __name__ == "__main__":
    main()
