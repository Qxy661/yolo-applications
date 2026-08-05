"""
数据 EDA（Exploratory Data Analysis）

分析数据集的分布特性：
  - 每张图数量分布
  - 钢珠像素尺寸分布（小目标分析）
  - 反光情况（高光点占比）
  - 位置分布（钢珠在画面中的位置）

Usage:
    python scripts/steelball_eda.py
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 字体配置：统一英文标签，用默认 DejaVu Sans（避免中文字体乱码）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = "/root/yolo-m2/steelball_data"


def load_labels(split="train"):
    """加载 YOLO 标注."""
    lbl_dir = os.path.join(DATA, "labels", split)
    img_dir = os.path.join(DATA, "images", split)
    samples = []
    for lbl in sorted(os.listdir(lbl_dir)):
        if not lbl.endswith(".txt"):
            continue
        with open(os.path.join(lbl_dir, lbl)) as f:
            boxes = [line.strip().split() for line in f if line.strip()]
        img_name = lbl.replace(".txt", ".jpg")
        samples.append((img_name, boxes))
    return samples


def analyze():
    samples = load_labels("train")
    print(f"训练集: {len(samples)} 张图")

    n_balls = []          # 每图Count
    ball_sizes = []       # 钢珠归一化宽高
    positions = []        # 钢珠中心位置
    highlight_ratio = []  # 反光（高光点）占比

    for img_name, boxes in samples[:200]:  # 抽样 200 张
        n_balls.append(len(boxes))
        img_path = os.path.join(DATA, "images", "train", img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        for b in boxes:
            cls, xc, yc, w, h = map(float, b)
            ball_sizes.append((w, h))
            positions.append((xc, yc))
            # 反光检测：bbox 内高亮点占比
            x1 = int((xc - w / 2) * W)
            y1 = int((yc - h / 2) * H)
            x2 = int((xc + w / 2) * W)
            y2 = int((yc + h / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            roi = gray[y1:y2, x1:x2]
            if roi.size > 0:
                hl = (roi > 200).mean()
                highlight_ratio.append(hl)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 数量分布
    ax = axes[0, 0]
    ax.hist(n_balls, bins=range(0, max(n_balls) + 2), edgecolor="black")
    ax.set_title(f"Steel balls per image (mean {np.mean(n_balls):.1f})")
    ax.set_xlabel("Count"); ax.set_ylabel("Images")

    # 2. 钢珠尺寸分布
    ax = axes[0, 1]
    sizes = np.array(ball_sizes)
    ax.scatter(sizes[:, 0] * 640, sizes[:, 1] * 640, s=3, alpha=0.5)
    mean_w = float(sizes[:, 0].mean()) * 640
    mean_h = float(sizes[:, 1].mean()) * 640
    ax.set_title(f"Steel ball size (mean {mean_w:.0f}x{mean_h:.0f})")
    ax.set_xlabel("Width(px)"); ax.set_ylabel("Height(px)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    # 3. 位置分布
    ax = axes[1, 0]
    pos = np.array(positions)
    ax.hist2d(pos[:, 0], pos[:, 1], bins=20, cmap="hot")
    ax.set_title("Position distribution (normalized)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # 4. 反光占比
    ax = axes[1, 1]
    ax.hist(highlight_ratio, bins=30, edgecolor="black")
    ax.set_title(f"Highlight ratio distribution (mean {np.mean(highlight_ratio):.2f})")
    ax.set_xlabel("Highlight ratio"); ax.set_ylabel("Frequency")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/steelball_eda.png", dpi=150)
    print(f"EDA 图已保存: outputs/steelball_eda.png")

    print(f"\n统计:")
    print(f"  每图Count: 均值 {np.mean(n_balls):.1f}, 范围 {min(n_balls)}-{max(n_balls)}")
    print(f"  钢珠尺寸: 均值 {mean_w:.1f}x{mean_h:.1f}px (归一化 {float(sizes[:, 0].mean()):.3f})")
    print(f"  Highlight ratio: 均值 {np.mean(highlight_ratio):.2f} (反光是普遍特征)")


if __name__ == "__main__":
    analyze()
