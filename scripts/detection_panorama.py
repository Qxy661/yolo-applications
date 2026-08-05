"""
检测结果全景图生成（论文级多图拼接）

支持模式：
  --mode numbered  框上标"类别固定编号"(所有car=1, pedestrian=2…)，底部画图例条
  --mode label     框上直接写类别名（如 car/person）
  --mode none      只画框（最干净，单类别）

用法：
  python scripts/detection_panorama.py \
      --images "data/visdrone/VisDrone2019-DET-val/images/*.jpg" \
      --model weights/best.pt --mode numbered \
      --names "0:pedestrian,1:people,2:bicycle,3:car,4:van,5:truck,6:tricycle,7:awning-tricycle,8:bus,9:motor" \
      --out results/visdrone/detection_panorama.jpg
"""
import argparse
import glob
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO


def detect_clean(model, img, conf=0.25, iou_thresh=0.45):
    """显式 NMS 检测，返回 [x1,y1,x2,y2,conf,cls]."""
    results = model(img, conf=min(conf, 0.05), verbose=False)
    all_boxes = []
    for r in results:
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu()
            confs = r.boxes.conf.cpu()
            clss = r.boxes.cls.cpu()
            for i in range(len(xyxy)):
                all_boxes.append([*xyxy[i].tolist(), confs[i].item(), clss[i].item()])
    if not all_boxes:
        return []
    boxes_t = torch.tensor([b[:4] for b in all_boxes])
    scores_t = torch.tensor([b[4] for b in all_boxes])
    keep = nms(boxes_t, scores_t, iou_threshold=iou_thresh)
    return [all_boxes[i] for i in keep.tolist() if all_boxes[i][4] >= conf]


def build_num_map(all_boxes):
    """给出现的每个类别分配固定编号（按类别 id 升序）。
    这样所有 car 都标 1、所有 pedestrian 都标 2，多图全景的题注全局一致。"""
    ids = sorted({int(b[5]) for boxes in all_boxes for b in boxes})
    return {cid: i + 1 for i, cid in enumerate(ids)}


def draw_clean(img, boxes, mode, num_map, names, color=(0, 255, 0)):
    """精细绘制.
    mode: none=只框, numbered=类别固定编号, label=类别名
    """
    vis = img.copy()
    font_scale = 0.4
    line_width = 1

    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        cls_id = int(b[5]) if len(b) > 5 else 0
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_width)

        if mode == "none":
            continue
        elif mode == "numbered":
            label = str(num_map.get(cls_id, cls_id + 1))
        else:  # label 模式
            label = names.get(cls_id, f"c{cls_id}") if names else f"c{cls_id}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.putText(vis, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    return vis


def _wrap_entries(entries, canvas_w, font_scale, thickness=1, pad=12):
    """把图例条目按画布宽度折行，返回 (行列表, 行高)."""
    rows, cur = [], ""
    for e in entries:
        trial = e if not cur else f"{cur}  {e}"
        (tw, th), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if cur and tw > canvas_w - 2 * pad:
            rows.append(cur)
            cur = e
        else:
            cur = trial
    if cur:
        rows.append(cur)
    return rows, th


def draw_legend(canvas, num_map, names):
    """底部画图例条：1=car  2=pedestrian ...（自包含，可随处复用）."""
    entries = [f"{n}={names.get(cid, f'c{cid}')}"
               for cid, n in sorted(num_map.items(), key=lambda kv: kv[1])]
    h, w = canvas.shape[:2]
    font_scale = 0.5 if len(entries) <= 6 else 0.42
    rows, th = _wrap_entries(entries, w, font_scale)
    bar_h = len(rows) * (th + 6) + 14
    bar = np.ones((bar_h, w, 3), dtype=np.uint8) * 250
    cv2.rectangle(bar, (0, 0), (w - 1, bar_h - 1), (0, 0, 0), 1)
    y = 12 + th
    for row in rows:
        (tw, th), _ = cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x = max(8, (w - tw) // 2)
        cv2.putText(bar, row, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (20, 20, 20), 1, cv2.LINE_AA)
        y += th + 6
    return np.vstack([canvas, bar])


def _fit_cover(img, cell_w, cell_h):
    """等比缩放到填满格子，再居中裁剪（cover）——横竖图统一、无留白."""
    h, w = img.shape[:2]
    scale = max(cell_w / w, cell_h / h)  # cover：按大的缩放
    nw, nh = max(cell_w, int(w * scale)), max(cell_h, int(h * scale))
    img = cv2.resize(img, (nw, nh))
    # 居中裁剪
    x = (nw - cell_w) // 2
    y = (nh - cell_h) // 2
    return img[y:y + cell_h, x:x + cell_w]


def make_panorama(model, images, mode, names, color, cols=3, max_w=480,
                  cell_w=None, cell_h=None):
    """绘制多张检测图并拼网格，返回 (画布, 类别编号映射).
    cell_w/cell_h: 若指定则每格统一尺寸(cover裁剪)，否则自适应."""
    det_results = []  # (img, boxes)
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        boxes = detect_clean(model, img)
        det_results.append((img, boxes))

    if not det_results:
        raise RuntimeError("No valid images found (glob matched nothing readable).")

    num_map = build_num_map([b for _, b in det_results])
    drawn = [draw_clean(img, boxes, mode, num_map, names, color)
             for img, boxes in det_results]

    if cell_w and cell_h:
        # 统一格子尺寸（cover 裁剪，无留白）
        drawn = [_fit_cover(d, cell_w, cell_h) for d in drawn]
        cell_w = cell_w
        cell_h = cell_h
    else:
        cell_h = max(d.shape[0] for d in drawn)
        cell_w = max(d.shape[1] for d in drawn)

    rows = (len(drawn) + cols - 1) // cols
    canvas = np.ones((cell_h * rows, cell_w * cols, 3), dtype=np.uint8) * 255
    for i, img in enumerate(drawn):
        r, c = i // cols, i % cols
        canvas[r * cell_h:r * cell_h + cell_h, c * cell_w:c * cell_w + cell_w] = img

    if mode == "numbered" and names and num_map:
        canvas = draw_legend(canvas, num_map, names)
    return canvas, num_map


def parse_names(s):
    """解析 "0:car,1:person" -> {0:'car', 1:'person'}."""
    if not s:
        return None
    d = {}
    for item in s.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            d[int(k)] = v.strip()
    return d


VISDRONE_NAMES = ("0:pedestrian,1:people,2:bicycle,3:car,4:van,5:truck,"
                  "6:tricycle,7:awning-tricycle,8:bus,9:motor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="numbered", choices=["none", "numbered", "label"])
    parser.add_argument("--names", default=None, help="0:car,1:person")
    parser.add_argument("--color", default="green", choices=["green", "orange", "blue"])
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--max-w", type=int, default=640, help="每格缩放宽度px")
    parser.add_argument("--cell-w", type=int, default=None, help="固定格子宽度px(统一尺寸)")
    parser.add_argument("--cell-h", type=int, default=None, help="固定格子高度px(统一尺寸)")
    parser.add_argument("--out", default="panorama.jpg")
    args = parser.parse_args()

    color = {"green": (0, 255, 0), "orange": (0, 165, 255), "blue": (255, 0, 0)}[args.color]
    names = parse_names(args.names or VISDRONE_NAMES)
    if args.images.endswith(".txt"):
        # 文件列表：每行一个图片路径
        images = [l.strip() for l in open(args.images) if l.strip()]
    else:
        images = sorted(glob.glob(args.images))
    print(f"匹配 {len(images)} 张图片，取前 {min(len(images), 9)} 张")
    model = YOLO(args.model)
    canvas, num_map = make_panorama(model, images[:9], args.mode, names, color,
                                    args.cols, args.max_w, args.cell_w, args.cell_h)
    cv2.imwrite(args.out, canvas)
    print(f"全景图已保存: {args.out}")

    if args.mode == "numbered" and num_map:
        legend = "  ".join(f"{n}={names.get(c, f'c{c}')}"
                           for c, n in sorted(num_map.items(), key=lambda kv: kv[1]))
        print("图例（类别固定编号）:")
        print(f"  {legend}")


if __name__ == "__main__":
    main()
