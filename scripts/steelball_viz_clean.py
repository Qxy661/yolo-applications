"""
钢珠检测图打磨脚本（论文级可视化）

解决检测图常见问题：
1. 框太密 → 显式 NMS + 细线框
2. 框太近 → iou 调优
3. 字体太大 → font_size 缩小
4. 标签挤一起 → 不显示 score（只显示标签）

Usage:
    python scripts/steelball_viz_clean.py --image test.jpg
"""
import argparse
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO


def detect_clean(model, img, conf=0.25, iou_thresh=0.45):
    """显式 NMS 检测（低置信度检测 + NMS 合并，解决重复框）."""
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


def draw_clean(img, boxes):
    """精细绘制：细框 + 小字体 + 不显示 score."""
    vis = img.copy()
    h, w = img.shape[:2]
    # 动态字体大小（小图用更小字体）
    font_scale = 0.4 if min(h, w) > 400 else 0.3
    line_width = 1 if min(h, w) > 400 else 1

    for b in boxes:
        x1, y1, x2, y2, c = [int(v) if i < 4 else v for i, v in enumerate(b)]
        # 细框（绿色）
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), line_width)
        # 标签放框外上方（不显示score，避免遮挡）
        label = "steel_ball"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # 标签背景（半透明，提高可读性）
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 2, y1), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.putText(vis, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="/mnt/e/yolo-visdrone/runs/steelball/steelball_yolo26/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", default="steelball_clean.png")
    args = parser.parse_args()

    model = YOLO(args.model)
    img = cv2.imread(args.image)
    boxes = detect_clean(model, img, args.conf, args.iou)
    vis = draw_clean(img, boxes)
    cv2.imwrite(args.out, vis)
    print(f"打磨完成: {args.out}, {len(boxes)} 个钢珠")
    print(f"置信度: {[round(b[4], 2) for b in boxes]}")


if __name__ == "__main__":
    main()
