"""SAHI evaluation for P2+CBAM model."""
import sys
from pathlib import Path

# Register custom modules
sys.path.insert(0, str(Path(__file__).parent))
import register_custom_modules

import json
import time
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


def build_coco_gt(val_image_dir, val_label_dir):
    coco = {"images": [], "annotations": [], "categories": []}
    for i, name in enumerate(VISDRONE_CLASSES):
        coco["categories"].append({"id": i, "name": name})
    ann_id = 0
    img_files = sorted(Path(val_image_dir).glob("*.jpg"))
    for img_id, img_file in enumerate(img_files):
        img = Image.open(img_file)
        w, h = img.size
        coco["images"].append({"id": img_id, "file_name": img_file.name, "width": w, "height": h})
        label_file = Path(val_label_dir) / (img_file.stem + ".txt")
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x = (x_c - bw / 2) * w
                    y = (y_c - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": cls_id,
                        "bbox": [x, y, box_w, box_h], "area": box_w * box_h, "iscrowd": 0
                    })
                    ann_id += 1
    return coco


def main():
    project_root = Path(__file__).parent.parent
    val_image_dir = project_root / "data" / "visdrone" / "images" / "val"
    val_label_dir = project_root / "data" / "visdrone" / "labels" / "val"
    weights_path = project_root / "runs" / "p2" / "yolov8s_p2_cbam" / "weights" / "best.pt"

    print("=" * 60)
    print("  SAHI Evaluation - P2+CBAM")
    print("=" * 60)

    # 1. Build COCO GT
    print("\n[1/4] Building COCO GT...")
    coco_gt_dict = build_coco_gt(val_image_dir, val_label_dir)
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    print("  Images: %d, Annotations: %d" % (len(coco_gt_dict['images']), len(coco_gt_dict['annotations'])))

    # 2. Load model
    print("\n[2/4] Loading SAHI model...")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=0.05,
        device="cuda:0",
    )
    print("  Model: %s" % weights_path.name)

    # 3. SAHI inference
    print("\n[3/4] SAHI slicing inference...")
    coco_results = []
    img_files = sorted(val_image_dir.glob("*.jpg"))
    start_time = time.time()

    for img_id, img_file in enumerate(tqdm(img_files, desc="Inference")):
        result = get_sliced_prediction(
            image=str(img_file),
            detection_model=model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=True,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_coco_bbox()
            coco_results.append({
                "image_id": img_id,
                "category_id": pred.category.id,
                "bbox": bbox,
                "score": pred.score.value,
            })

    elapsed = time.time() - start_time
    print("  Time: %.1fs (%.2fs/image)" % (elapsed, elapsed / len(img_files)))
    print("  Predictions: %d" % len(coco_results))

    # 4. Evaluate
    print("\n[4/4] COCO mAP evaluation...")
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    prec = coco_eval.eval["precision"]
    per_class_ap = {}
    for i, name in enumerate(VISDRONE_CLASSES):
        ap50_slice = prec[0, :, i, 0, 2]
        valid = ap50_slice[ap50_slice > -1]
        ap50 = float(valid.mean()) if len(valid) > 0 else 0.0
        per_class_ap[name] = round(ap50, 4)

    results = {
        "method": "SAHI + P2+CBAM",
        "mAP50": float(stats[1]),
        "mAP50_95": float(stats[0]),
        "total_predictions": len(coco_results),
        "inference_time_s": round(elapsed, 1),
        "per_class_ap50": per_class_ap,
    }

    print("\n" + "=" * 60)
    print("  SAHI + P2+CBAM Results")
    print("=" * 60)
    print("  mAP@0.5:    %.4f" % results['mAP50'])
    print("  mAP@0.5:0.95: %.4f" % results['mAP50_95'])
    print("\n  Per-class AP@0.5:")
    for name, ap in per_class_ap.items():
        print("    %-20s: %.4f" % (name, ap))

    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "sahi_p2_cbam_metrics.json"
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n  Saved: %s" % output_file)

    return results


if __name__ == "__main__":
    main()
