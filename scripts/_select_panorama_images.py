"""
临时脚本：按 VisDrone 典型难点特征挑选 9 张全景展示图。

特征维度（针对 README 里"漏检为主/小目标典型问题"）：
  D = 密度：目标总面积 / 图像面积（拥挤度）
  S = 小目标：目标平均像素面积（越小越典型）
  N = 多目标：检测框数
  C = 遮挡/复杂：框间重叠面积占比 + 类别数
综合分 = 权重加权，选出 9 张最能代表"密/小/多/遮挡"的图。
输出：选中的图片路径写入 /tmp/panorama_sel.txt
"""
import sys, glob, os
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_panorama import detect_clean, YOLO

IMGDIR = "/mnt/e/yolo-visdrone/data/visdrone/VisDrone2019-DET-val/images"
MODEL = "/mnt/e/yolo-visdrone/weights/best.pt"

model = YOLO(MODEL)
paths = sorted(glob.glob(os.path.join(IMGDIR, "*.jpg")))
sampled = paths[::6]  # 抽样 ~1/6 提速
print(f"共 {len(paths)} 张，抽样 {len(sampled)} 张检测中...")


def overlap_ratio(boxes):
    """框间重叠面积 / 总框面积 —— 衡量遮挡/拥挤."""
    if len(boxes) < 2:
        return 0.0
    total = 0.0
    area_sum = 0.0
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i][:4]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        area_sum += area
        for j in range(i + 1, len(boxes)):
            a1, b1, a2, b2 = boxes[j][:4]
            ix1, iy1 = max(x1, a1), max(y1, b1)
            ix2, iy2 = min(x2, a2), min(y2, b2)
            if ix2 > ix1 and iy2 > iy1:
                total += (ix2 - ix1) * (iy2 - iy1)
    return total / area_sum if area_sum > 0 else 0.0


scored = []
for p in sampled:
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    img_area = h * w
    boxes = detect_clean(model, img)
    if not boxes:
        continue
    n = len(boxes)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    # 维度（都归一化到 0~1 越高越好）
    density = sum(areas) / img_area                       # 拥挤度
    small = 1 - min(1.0, sum(areas) / n / 2500)          # 平均目标小(2500px=50x50上限)
    many = min(1.0, n / 25)                              # 目标多
    occl = overlap_ratio(boxes) + 0.05 * len({int(b[5]) for b in boxes})  # 遮挡+类别
    # 加权：密度+小目标为主
    score = 3.0 * density + 2.5 * small + 2.0 * many + 1.5 * min(occl, 1.0)
    scored.append((score, p, n, density, small, occl))

scored.sort(key=lambda x: -x[0])
# 去重：避免选到同一场景的相邻帧
picked = []
last_prefix = ""
for s, p, n, d, sm, oc in scored:
    prefix = os.path.basename(p)[:12]  # 场景前缀
    if prefix == last_prefix and len(picked) >= 5:
        continue
    if len(picked) >= 9:
        break
    picked.append((p, n, d, sm, oc))
    last_prefix = prefix

with open("/tmp/panorama_sel.txt", "w") as f:
    for p, *_ in picked:
        f.write(p + "\n")
print(f"\n选中 {len(picked)} 张 -> /tmp/panorama_sel.txt")
for p, n, d, sm, oc in picked:
    print(f"  {os.path.basename(p)}  boxes={n:2d}  density={d*100:4.1f}%  small={1-sm:4.0f}px  occl={oc*100:4.1f}%")
