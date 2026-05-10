"""
工具函数
"""
import os
import random
import cv2
import numpy as np
from pathlib import Path


# VisDrone 类别名
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# 类别颜色 (BGR)
CLASS_COLORS = [
    (255, 0, 0),     # pedestrian - 蓝
    (0, 255, 0),     # people - 绿
    (0, 0, 255),     # bicycle - 红
    (255, 255, 0),   # car - 青
    (255, 0, 255),   # van - 品红
    (0, 255, 255),   # truck - 黄
    (128, 0, 0),     # tricycle - 深蓝
    (0, 128, 0),     # awning-tricycle - 深绿
    (0, 0, 128),     # bus - 深红
    (128, 128, 0),   # motor - 深青
]


def yolo_to_xyxy(bbox, img_w, img_h):
    """YOLO 归一化格式 → 像素坐标 [x1,y1,x2,y2]"""
    x_c, y_c, w, h = bbox
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return [x1, y1, x2, y2]


def draw_boxes(img, labels, class_names=None, colors=None):
    """在图片上绘制 YOLO 标注框"""
    if class_names is None:
        class_names = CLASS_NAMES
    if colors is None:
        colors = CLASS_COLORS

    h, w = img.shape[:2]
    for label in labels:
        parts = label.strip().split()
        cls_id = int(parts[0])
        bbox = [float(x) for x in parts[1:5]]
        x1, y1, x2, y2 = yolo_to_xyxy(bbox, w, h)

        color = colors[cls_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def visualize_samples(data_dir: str, split: str = "train", n_samples: int = 10, output_dir: str = "results/samples"):
    """随机可视化数据集样本"""
    data_path = Path(data_dir)
    img_dir = data_path / "images" / split
    lbl_dir = data_path / "labels" / split
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    samples = random.sample(img_files, min(n_samples, len(img_files)))

    for img_file in samples:
        img = cv2.imread(str(img_file))
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            labels = lbl_file.read_text().strip().split('\n')
            img = draw_boxes(img, [l for l in labels if l.strip()])

        save_path = out_path / img_file.name
        cv2.imwrite(str(save_path), img)
        print(f"  {save_path.name}")


def count_labels(data_dir: str, split: str = "train"):
    """统计各类别目标数量"""
    data_path = Path(data_dir)
    lbl_dir = data_path / "labels" / split

    counts = [0] * len(CLASS_NAMES)
    total_files = 0

    for lbl_file in lbl_dir.glob("*.txt"):
        total_files += 1
        for line in lbl_file.read_text().strip().split('\n'):
            if not line.strip():
                continue
            cls_id = int(line.split()[0])
            if 0 <= cls_id < len(CLASS_NAMES):
                counts[cls_id] += 1

    print(f"\n数据集统计 ({split}):")
    print(f"  文件数: {total_files}")
    print(f"  目标总数: {sum(counts)}")
    for i, (name, cnt) in enumerate(zip(CLASS_NAMES, counts)):
        pct = cnt / sum(counts) * 100 if sum(counts) > 0 else 0
        print(f"  {i:2d} {name:20s}: {cnt:6d} ({pct:5.1f}%)")

    return counts


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/visdrone"

    count_labels(data_dir, "train")
    count_labels(data_dir, "val")
