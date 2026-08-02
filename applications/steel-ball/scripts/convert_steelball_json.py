"""
Convert steel-ball dataset: LabelMe json -> YOLO txt + data cleaning.

Source: daffae/steelball_detect (jsonlabels/ has LabelMe format, 1851 files;
txtlabels/ has 430 YOLO files). We merge both into a clean YOLO dataset.

LabelMe json structure:
    { "imagePath": "...", "imageWidth": W, "imageHeight": H,
      "shapes": [ {"label": "ball", "shape_type": "rectangle",
                   "points": [[x1,y1],[x2,y2]]} ] }

Cleaning steps:
  1. Convert LabelMe rectangle -> YOLO (class xc yc w h), normalized
  2. Filter invalid boxes (out of bounds / zero area)
  3. Dedup: if a json-derived label already has a txt, prefer the txt
     (assume txt is the curated one)
  4. Split train/val (80/20)
"""
import json
import os
import random
import shutil

SRC = "/root/yolo-m2/steelball_detect"
DST = "/root/yolo-m2/steelball_data"
VAL_RATIO = 0.2
SEED = 42

CLASS = 0  # steel_ball


def labelme_to_yolo(shapes, img_w, img_h):
    """Convert LabelMe shapes to YOLO lines. Returns list of valid lines."""
    lines = []
    for s in shapes:
        if s.get("shape_type") != "rectangle":
            continue
        label = s.get("label", "").lower()
        if "ball" not in label and "steel" not in label:
            continue  # only ball/steel_ball class
        pts = s["points"]
        if len(pts) != 2:
            continue
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue  # invalid box
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        wn = w / img_w
        hn = h / img_h
        # Filter out-of-bounds (allow tiny epsilon)
        if xc < 0 or xc > 1 or yc < 0 or yc > 1 or wn <= 0 or hn <= 0:
            continue
        lines.append(f"{CLASS} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return lines


def main():
    random.seed(SEED)
    os.makedirs(f"{DST}/images/train", exist_ok=True)
    os.makedirs(f"{DST}/images/val", exist_ok=True)
    os.makedirs(f"{DST}/labels/train", exist_ok=True)
    os.makedirs(f"{DST}/labels/val", exist_ok=True)

    # Collect all image names that have a usable annotation
    images = set(os.listdir(f"{SRC}/images"))
    jsons = os.listdir(f"{SRC}/jsonlabels")
    txts = os.listdir(f"{SRC}/txtlabels")

    usable = []
    stats = {"from_json": 0, "from_txt": 0, "invalid": 0}

    for img in sorted(images):
        stem = img.rsplit(".", 1)[0]
        yolo_lines = None

        # 1. Prefer existing txt label (curated)
        txt = f"{stem}.txt"
        if txt in txts:
            p = os.path.join(f"{SRC}/txtlabels", txt)
            with open(p) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                yolo_lines = lines
                stats["from_txt"] += 1

        # 2. Else try json (LabelMe)
        if yolo_lines is None:
            jf = f"{stem}.json"
            if jf in jsons:
                with open(os.path.join(f"{SRC}/jsonlabels", jf)) as f:
                    data = json.load(f)
                w, h = data.get("imageWidth", 0), data.get("imageHeight", 0)
                if w and h:
                    lines = labelme_to_yolo(data.get("shapes", []), w, h)
                    if lines:
                        yolo_lines = lines
                        stats["from_json"] += 1
                    else:
                        stats["invalid"] += 1
                else:
                    stats["invalid"] += 1

        if yolo_lines:
            usable.append((img, yolo_lines))

    # Split
    random.shuffle(usable)
    n_val = int(len(usable) * VAL_RATIO)

    for i, (img, lines) in enumerate(usable):
        split = "val" if i < n_val else "train"
        stem = img.rsplit(".", 1)[0]
        shutil.copy(os.path.join(f"{SRC}/images", img),
                    f"{DST}/images/{split}/{img}")
        with open(f"{DST}/labels/{split}/{stem}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Total usable: {len(usable)}")
    print(f"  from json: {stats['from_json']}, from txt: {stats['from_txt']}, invalid: {stats['invalid']}")
    print(f"train: {len(os.listdir(f'{DST}/images/train'))} 图")
    print(f"val: {len(os.listdir(f'{DST}/images/val'))} 图")


if __name__ == "__main__":
    main()
