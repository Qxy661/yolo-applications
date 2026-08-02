"""SAHI 切片推理评估（新 best.pt，M2 收尾1 优化对比）"""
import json
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

WEIGHTS = "/mnt/e/yolo-visdrone/runs/yolo26/yolo26s_full2/weights/best.pt"
VAL_IMG = "/root/yolo-m2/data/images/val"
VAL_LBL = "/root/yolo-m2/data/labels/val"
SLICE_SIZE = 640
SLICE_OVERLAP = 0.3


def build_coco_gt(img_dir, lbl_dir):
    coco = {"images": [], "annotations": [], "categories": []}
    for i, name in enumerate(VISDRONE_CLASSES):
        coco["categories"].append({"id": i, "name": name})
    ann_id = 0
    for img_id, img_file in enumerate(sorted(Path(img_dir).glob("*.jpg"))):
        img = Image.open(img_file)
        w, h = img.size
        coco["images"].append({"id": img_id, "file_name": img_file.name, "width": w, "height": h})
        lbl = Path(lbl_dir) / (img_file.stem + ".txt")
        if lbl.exists():
            for line in open(lbl):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])
                x_px, y_px = (x - bw / 2) * w, (y - bh / 2) * h
                bw_px, bh_px = bw * w, bh * h
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cls,
                    "bbox": [x_px, y_px, bw_px, bh_px], "area": bw_px * bh_px,
                    "iscrowd": 0,
                })
                ann_id += 1
    return coco


def main():
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=WEIGHTS,
        confidence_threshold=0.2, image_size=800,
    )
    print(f"模型: {Path(WEIGHTS).name}")
    img_files = sorted(Path(VAL_IMG).glob("*.jpg"))
    print(f"验证图: {len(img_files)} 张")
    coco_gt = build_coco_gt(VAL_IMG, VAL_LBL)

    coco_results = []
    for img_file in tqdm(img_files, desc="SAHI 切片推理"):
        result = get_sliced_prediction(
            str(img_file), model,
            slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
            overlap_height_ratio=SLICE_OVERLAP, overlap_width_ratio=SLICE_OVERLAP,
        )
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xywh()
            coco_results.append({
                "image_id": img_files.index(img_file),
                "category_id": pred.category.id,
                "bbox": bbox,
                "score": float(pred.score.value),
            })

    print(f"检测框: {len(coco_results)}")

    coco_gt_obj = COCO()
    coco_gt_obj.dataset = coco_gt
    coco_gt_obj.createIndex()
    coco_dt_obj = coco_gt_obj.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP50 = coco_eval.stats[1]  # AP@IoU=0.50 (stats[1])
    mAP50_95 = coco_eval.stats[0]  # AP@IoU=0.50:0.95 (stats[0])

    prec = coco_eval.eval["precision"]
    per_class_ap = {}
    for i, name in enumerate(VISDRONE_CLASSES):
        ap50_slice = prec[0, :, i, 0, 2]
        valid = ap50_slice[ap50_slice > -1]
        per_class_ap[name] = round(float(valid.mean()), 4) if len(valid) > 0 else 0.0

    results = {
        "method": "SAHI (new best.pt)",
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "per_class_ap50": per_class_ap,
    }
    print("\n=== SAHI 评估结果 ===")
    print(f"  mAP@0.5:    {mAP50:.4f}")
    print(f"  mAP@0.5:0.95: {mAP50_95:.4f}")
    print("  各类别 AP@0.5:")
    for name, ap in per_class_ap.items():
        print(f"    {name:20s}: {ap:.4f}")

    out = Path("/mnt/e/yolo-visdrone/results/sahi_new_metrics.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
