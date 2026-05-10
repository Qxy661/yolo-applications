"""
VisDrone2019-DET 数据集下载与转换脚本
将 VisDrone 标注格式转换为 YOLO 格式
"""
import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm

# VisDrone 下载链接 (GitHub Release 镜像)
DATASET_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test.zip",
}

# 类别映射 (VisDrone 原始编号 → YOLO 从0开始)
# VisDrone: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
#           6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor
VISDRONE_TO_YOLO = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}


def download_file(url: str, dest: str):
    """下载文件，显示进度条"""
    import urllib.request
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  已存在: {dest.name}")
        return
    print(f"  下载中: {dest.name}")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=dest.name) as t:
        def reporthook(block_num, block_size, total_size):
            t.total = total_size
            t.update(block_num * block_size - t.n)
        urllib.request.urlretrieve(url, str(dest), reporthook=reporthook)


def extract_zip(zip_path: str, extract_to: str):
    """解压 zip 文件"""
    print(f"  解压中: {Path(zip_path).name}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)


def convert_visdrone_to_yolo(anno_dir: str, img_dir: str, output_label_dir: str):
    """
    将 VisDrone 标注转换为 YOLO 格式

    VisDrone 标注格式 (每行):
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

    YOLO 格式 (每行):
        <class> <x_center> <y_center> <width> <height>  (归一化到 0-1)
    """
    anno_path = Path(anno_dir)
    img_path = Path(img_dir)
    out_path = Path(output_label_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    anno_files = list(anno_path.glob("*.txt"))
    print(f"  转换标注: {len(anno_files)} 个文件")

    for af in tqdm(anno_files, desc="转换标注"):
        # 读取图片尺寸
        img_file = img_path / (af.stem + ".jpg")
        if not img_file.exists():
            continue

        from PIL import Image
        with Image.open(img_file) as im:
            img_w, img_h = im.size

        # 读取标注
        lines = af.read_text().strip().split('\n')
        yolo_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue

            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            score = float(parts[4])      # -1 for train/val, 1 for test
            cls_id = int(parts[5])

            # 跳过 ignored (0) 和 score < 0 的
            if cls_id == 0 or score < 0:
                continue
            if cls_id not in VISDRONE_TO_YOLO:
                continue

            # 转换为 YOLO 归一化格式
            cls_yolo = VISDRONE_TO_YOLO[cls_id]
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # 限幅到 [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            yolo_lines.append(f"{cls_yolo} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # 写入 YOLO 标注
        out_file = out_path / (af.stem + ".txt")
        out_file.write_text('\n'.join(yolo_lines))


def organize_yolo_structure(data_dir: str):
    """整理为 YOLO 标准目录结构"""
    data_path = Path(data_dir)

    for split in ["train", "val", "test"]:
        src_img = data_path / f"VisDrone2019-DET-{split}" / "images"
        src_anno = data_path / f"VisDrone2019-DET-{split}" / "annotations"
        dst_img = data_path / "images" / split
        dst_lbl = data_path / "labels" / split

        if not src_img.exists():
            print(f"  [跳过] {split}: 源目录不存在 {src_img}")
            continue

        # 创建目标目录
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # 复制/链接图片
        img_files = list(src_img.glob("*.jpg"))
        print(f"  [{split}] 复制 {len(img_files)} 张图片...")
        for img in tqdm(img_files, desc=f"复制图片-{split}"):
            dst = dst_img / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

        # 转换标注
        if src_anno.exists():
            print(f"  [{split}] 转换标注...")
            convert_visdrone_to_yolo(str(src_anno), str(src_img), str(dst_lbl))
        else:
            print(f"  [{split}] 无标注目录 (test set)")


def main():
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "visdrone"
    zip_dir = data_dir / "zips"

    print("=" * 50)
    print("  VisDrone2019-DET 数据集下载与转换")
    print("=" * 50)

    # 1. 下载
    print("\n[Step 1/3] 下载数据集...")
    for split, url in DATASET_URLS.items():
        zip_file = zip_dir / f"VisDrone2019-DET-{split}.zip"
        download_file(url, str(zip_file))

    # 2. 解压
    print("\n[Step 2/3] 解压数据集...")
    for split in DATASET_URLS:
        zip_file = zip_dir / f"VisDrone2019-DET-{split}.zip"
        if zip_file.exists():
            extract_zip(str(zip_file), str(data_dir))

    # 3. 转换格式
    print("\n[Step 3/3] 转换为 YOLO 格式...")
    organize_yolo_structure(str(data_dir))

    # 统计
    print("\n" + "=" * 50)
    print("  数据集准备完成！")
    print("=" * 50)
    for split in ["train", "val"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split}: {n_img} 图片, {n_lbl} 标注")

    print(f"\n  配置文件: {data_dir / 'visdrone.yaml'}")
    print("  开始训练: python src/train.py")


if __name__ == "__main__":
    main()
