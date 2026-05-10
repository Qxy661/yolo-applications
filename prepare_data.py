"""
解压 VisDrone 并转换为 YOLO 格式
"""
import zipfile
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

VISDRONE_TO_YOLO = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}

def extract_zip(zip_path, dest):
    print(f"  解压 {Path(zip_path).name} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)

def convert_labels(anno_dir, img_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for af in tqdm(sorted(Path(anno_dir).glob("*.txt")), desc="转换标注"):
        img_file = Path(img_dir) / (af.stem + ".jpg")
        if not img_file.exists():
            continue
        with Image.open(img_file) as im:
            iw, ih = im.size
        lines = []
        for line in af.read_text().strip().split('\n'):
            if not line.strip():
                continue
            p = line.split(',')
            if len(p) < 6:
                continue
            x, y, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            cls = int(p[5])
            if cls == 0 or cls not in VISDRONE_TO_YOLO:
                continue
            c = VISDRONE_TO_YOLO[cls]
            xc = max(0, min(1, (x + w/2) / iw))
            yc = max(0, min(1, (y + h/2) / ih))
            wn = max(0, min(1, w / iw))
            hn = max(0, min(1, h / ih))
            lines.append(f"{c} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        (out / (af.stem + ".txt")).write_text('\n'.join(lines))

def main():
    root = Path(__file__).parent
    data = root / "data" / "visdrone"
    zips = data / "zips"

    for split in ["train", "val"]:
        zf = zips / f"VisDrone2019-DET-{split}.zip"
        if zf.exists():
            extract_zip(str(zf), str(data))

    for split in ["train", "val"]:
        src = data / f"VisDrone2019-DET-{split}"
        if not src.exists():
            print(f"  [跳过] {split}")
            continue
        dst_img = data / "images" / split
        dst_lbl = data / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # 复制图片
        imgs = sorted((src / "images").glob("*.jpg"))
        print(f"  [{split}] 复制 {len(imgs)} 张图片...")
        for img in tqdm(imgs, desc=f"复制-{split}"):
            d = dst_img / img.name
            if not d.exists():
                shutil.copy2(str(img), str(d))

        # 转换标注
        anno = src / "annotations"
        if anno.exists():
            print(f"  [{split}] 转换标注...")
            convert_labels(str(anno), str(src / "images"), str(dst_lbl))

    # 统计
    for split in ["train", "val"]:
        n_img = len(list((data / "images" / split).glob("*.jpg")))
        n_lbl = len(list((data / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_img} 图片, {n_lbl} 标注")

    print("\n数据集准备完成!")

if __name__ == "__main__":
    main()
