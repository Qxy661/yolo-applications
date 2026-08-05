"""
临时脚本：按钢珠场景特征精选 9 张全景展示图。

钢珠数据集有两个典型工业场景：
  1. 密集多球（ball_weixin_*）：球多、球大、暗背景 —— 体现检测数量能力
  2. 高光小目标（ball_2026*）：球少、球小、高光明显 —— 体现反光指纹特征

选图策略：两类各选一部分，密度类选球最多最密的，高光类选高光最强球最小的。
输出：/tmp/panorama_steelball.txt
"""
import sys, glob, os
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

IMGDIR = "/root/yolo-m2/steelball_data/images/val"
MODEL = "/mnt/e/yolo-visdrone/applications/steel-ball/results/best.pt"

model = YOLO(MODEL)


def detect(img, conf=0.25, iou=0.45):
    r = model(img, conf=min(conf, 0.05), verbose=False)
    ab = []
    for rr in r:
        if rr.boxes is not None:
            xy = rr.boxes.xyxy.cpu(); cf = rr.boxes.conf.cpu()
            for i in range(len(xy)):
                ab.append([*xy[i].tolist(), cf[i].item()])
    if not ab:
        return []
    keep = nms(torch.tensor([b[:4] for b in ab]),
               torch.tensor([b[4] for b in ab]), iou)
    return [ab[i] for i in keep.tolist() if ab[i][4] >= conf]


def analyze(p):
    img = cv2.imread(p)
    if img is None:
        return None
    bs = detect(img)
    if not bs:
        return None
    n = len(bs)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bs]
    avg_ball = np.mean(areas)
    bright = img.mean()
    hl = (img > 240).mean() * 100  # 高光占比
    return {"path": p, "n": n, "avg": avg_ball, "bright": bright, "hl": hl,
            "name": os.path.basename(p)}


paths = sorted(glob.glob(os.path.join(IMGDIR, "*.jpg")))
infos = [analyze(p) for p in paths]
infos = [x for x in infos if x]

# 分类：密集多球 vs 高光小目标
dense = [x for x in infos if x["name"].startswith("ball_weixin")]
glossy = [x for x in infos
          if not x["name"].startswith("ball_weixin") and x["n"] >= 5]

# 密集类：按球数降序取前 5
dense.sort(key=lambda x: -x["n"])
dense_sel = dense[:5]

# 高光类：按 高光占比*0.6 + 小目标(avg小)*0.4 综合分取前 5
def glossy_score(x):
    s_hl = min(x["hl"] / 10, 1.0)          # 高光占比 0~10% 归一
    s_small = 1 - min(x["avg"] / 3000, 1.0)  # 平均球面积小→分高
    return 0.6 * s_hl + 0.4 * s_small

glossy.sort(key=glossy_score, reverse=True)
glossy_sel = glossy[:5]

# 场景去重：weixin 密集系列直接全收；实拍系列按时间戳前缀去重
picked, seen_prefix = [], set()
for x in dense_sel + glossy_sel:
    if x["name"].startswith("ball_weixin"):
        picked.append(x)  # 密集系列每个都是独立场景
        continue
    prefix = x["name"][:21]  # 实拍系列按 ball_20260729_HHMMSS_XX 前缀去重(秒级+帧号)
    if prefix in seen_prefix:
        continue
    seen_prefix.add(prefix)
    picked.append(x)
    if len(picked) >= 9:
        break

with open("/tmp/panorama_steelball.txt", "w") as f:
    for x in picked:
        f.write(x["path"] + "\n")

print(f"选中 {len(picked)} 张 -> /tmp/panorama_steelball.txt")
print(f"{'文件':<30} {'球数':>4} {'平均px':>7} {'亮度':>5} {'高光':>6}")
for x in picked:
    print(f"{x['name']:<30} {x['n']:>4} {x['avg']:>7.0f} {x['bright']:>5.0f} {x['hl']:>5.1f}%")
print(f"\n场景分布: 密集多球 {sum(1 for x in picked if 'weixin' in x['name'])} 张, "
      f"高光小目标 {sum(1 for x in picked if 'weixin' not in x['name'])} 张")
