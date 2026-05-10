"""
零训练优化脚本
1. SAHI + TTA (测试时增强)
2. NMS/置信度阈值网格搜索
3. 多模型集成
"""
import json
import time
import os
import tempfile
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# Fix ultralytics PIL patch before importing YOLO
import PIL.Image
_pil_open = PIL.Image.open

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

# Restore original PIL.Image.open to avoid pi_heif error
PIL.Image.open = _pil_open

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def build_coco_gt(val_image_dir, val_label_dir):
    """构建 COCO ground truth"""
    coco = {"images": [], "annotations": [], "categories": []}
    for i, name in enumerate(VISDRONE_CLASSES):
        coco["categories"].append({"id": i, "name": name})

    ann_id = 0
    img_files = sorted(f for f in Path(val_image_dir).glob("*.jpg") if "_flip_temp" not in f.name)
    for img_id, img_file in enumerate(img_files):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": img_file.name, "width": w, "height": h})

        label_file = Path(val_label_dir) / (img_file.stem + ".txt")
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x = (x_c - bw / 2) * w
                    y = (y_c - bh / 2) * h
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": cls_id,
                        "bbox": [x, y, bw * w, bh * h], "area": bw * w * bh * h, "iscrowd": 0
                    })
                    ann_id += 1
    return coco


def evaluate_coco(coco_gt_dict, coco_results, method_name=""):
    """通用 COCO 评估"""
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    if len(coco_results) == 0:
        print(f"  {method_name}: no predictions!")
        return {}

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
        per_class_ap[name] = round(float(valid.mean()), 4) if len(valid) > 0 else 0.0

    return {
        "method": method_name,
        "mAP50": float(stats[1]),
        "mAP50_95": float(stats[0]),
        "total_predictions": len(coco_results),
        "per_class_ap50": per_class_ap,
    }


def nms_per_class(preds, iou_thresh=0.5):
    """Simple per-class NMS on COCO-format bboxes [x, y, w, h]"""
    from collections import defaultdict

    def iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2 = min(b1[0]+b1[2], b2[0]+b2[2])
        y2 = min(b1[1]+b1[3], b2[1]+b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        union = b1[2]*b1[3] + b2[2]*b2[3] - inter
        return inter / union if union > 0 else 0

    by_class = defaultdict(list)
    for p in preds:
        by_class[p["category_id"]].append(p)

    kept = []
    for cls, cls_preds in by_class.items():
        cls_preds.sort(key=lambda x: x["score"], reverse=True)
        while cls_preds:
            best = cls_preds.pop(0)
            kept.append(best)
            cls_preds = [p for p in cls_preds if iou(best["bbox"], p["bbox"]) < iou_thresh]
    return kept


def sahi_tta_predict(sahi_model, image_path, slice_size=640, overlap=0.2):
    """SAHI + TTA: original + horizontal flip, merge with NMS"""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    all_preds = []

    # Original
    result1 = get_sliced_prediction(
        image=str(image_path), detection_model=sahi_model,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        perform_standard_pred=True, postprocess_type="NMS",
        postprocess_match_threshold=0.5, verbose=0,
    )
    for pred in result1.object_prediction_list:
        bbox = pred.bbox.to_coco_bbox()
        all_preds.append({
            "category_id": pred.category.id, "bbox": bbox, "score": pred.score.value
        })

    # Horizontal flip
    img_flipped = cv2.flip(img, 1)
    fd, flipped_path = tempfile.mkstemp(suffix=".jpg", dir=str(Path(image_path).parent))
    os.close(fd)
    cv2.imwrite(flipped_path, img_flipped)

    result2 = get_sliced_prediction(
        image=flipped_path, detection_model=sahi_model,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        perform_standard_pred=True, postprocess_type="NMS",
        postprocess_match_threshold=0.5, verbose=0,
    )

    for pred in result2.object_prediction_list:
        bbox = pred.bbox.to_coco_bbox()
        x_orig = w - bbox[0] - bbox[2]
        all_preds.append({
            "category_id": pred.category.id,
            "bbox": [x_orig, bbox[1], bbox[2], bbox[3]],
            "score": pred.score.value
        })

    Path(flipped_path).unlink(missing_ok=True)

    # Apply NMS to merge overlapping detections from original + flipped
    return nms_per_class(all_preds, iou_thresh=0.5)


def run_sahi_tta(weights_path, val_image_dir, val_label_dir, output_file):
    """Experiment 1: SAHI + TTA"""
    print("\n" + "=" * 60)
    print("  Experiment 1: SAHI + TTA")
    print("=" * 60)

    coco_gt_dict = build_coco_gt(val_image_dir, val_label_dir)
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights_path),
        confidence_threshold=0.25, device="cuda:0",
    )

    img_files = sorted(f for f in val_image_dir.glob("*.jpg") if "_flip_temp" not in f.name)
    coco_results = []
    start = time.time()

    for img_id, img_file in enumerate(tqdm(img_files, desc="SAHI+TTA")):
        preds = sahi_tta_predict(sahi_model, img_file)
        for p in preds:
            p["image_id"] = img_id
            coco_results.append(p)

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(img_files):.2f}s/img)")

    result = evaluate_coco(coco_gt_dict, coco_results, "SAHI+TTA")
    result["inference_time_s"] = round(elapsed, 1)
    result["time_per_image_s"] = round(elapsed / len(img_files), 2)
    result["slice_size"] = [640, 640]
    result["overlap_ratio"] = 0.2
    result["tta"] = True

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_file}")
    return result


def run_threshold_search(sahi_model, val_image_dir, val_label_dir, img_files, coco_gt_dict, output_file):
    """Experiment 2: confidence threshold grid search"""
    print("\n" + "=" * 60)
    print("  Experiment 2: Threshold Grid Search")
    print("=" * 60)

    print("  Collecting predictions (conf=0.05)...")
    all_raw_preds = {}

    for img_id, img_file in enumerate(tqdm(img_files, desc="Collecting")):
        result = get_sliced_prediction(
            image=str(img_file), detection_model=sahi_model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            perform_standard_pred=True, postprocess_type="NMS",
            postprocess_match_threshold=0.5, verbose=0,
        )
        preds = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_coco_bbox()
            preds.append({
                "image_id": img_id, "category_id": pred.category.id,
                "bbox": bbox, "score": pred.score.value,
            })
        all_raw_preds[img_id] = preds

    conf_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    results_grid = []

    print("\n  Grid search...")
    for conf in conf_thresholds:
        filtered = []
        for img_id, preds in all_raw_preds.items():
            for p in preds:
                if p["score"] >= conf:
                    filtered.append(p)

        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        if len(filtered) == 0:
            continue
        coco_dt = coco_gt.loadRes(filtered)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if len(coco_eval.stats) == 0:
            continue
        map50 = float(coco_eval.stats[1])
        map50_95 = float(coco_eval.stats[0])
        results_grid.append({
            "conf_threshold": conf, "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4), "num_predictions": len(filtered),
        })
        print(f"    conf={conf:.2f}: mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}, preds={len(filtered)}")

    best = max(results_grid, key=lambda x: x["mAP50"])
    print(f"\n  Best: conf={best['conf_threshold']}, mAP@0.5={best['mAP50']:.4f}")

    output = {
        "method": "Threshold Grid Search (SAHI base)",
        "best_conf": best["conf_threshold"],
        "best_mAP50": best["mAP50"],
        "best_mAP50_95": best["mAP50_95"],
        "grid_results": results_grid,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_file}")
    return output


def run_ensemble(model1_path, model2_path, val_image_dir, val_label_dir, coco_gt_dict, output_file):
    """Experiment 3: multi-model ensemble"""
    print("\n" + "=" * 60)
    print("  Experiment 3: Multi-Model Ensemble")
    print("=" * 60)

    model1 = YOLO(str(model1_path))
    model2 = YOLO(str(model2_path))

    img_files = sorted(f for f in val_image_dir.glob("*.jpg") if "_flip_temp" not in f.name)
    coco_results = []
    start = time.time()

    for img_id, img_file in enumerate(tqdm(img_files, desc="Ensemble")):
        img_str = str(img_file)

        r1 = model1.predict(source=img_str, conf=0.15, iou=0.5, verbose=False)[0]
        r2 = model2.predict(source=img_str, conf=0.15, iou=0.5, verbose=False)[0]

        for box in r1.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            coco_results.append({
                "image_id": img_id, "category_id": int(box.cls[0]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(box.conf[0]) * 0.7,
            })
        for box in r2.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            coco_results.append({
                "image_id": img_id, "category_id": int(box.cls[0]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(box.conf[0]) * 1.0,
            })

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(img_files):.2f}s/img)")

    result = evaluate_coco(coco_gt_dict, coco_results, "Ensemble (v8n*0.7 + v8s*1.0)")
    result["inference_time_s"] = round(elapsed, 1)
    result["time_per_image_s"] = round(elapsed / len(img_files), 2)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_file}")
    return result


def main():
    project_root = Path(__file__).parent.parent
    val_image_dir = project_root / "data" / "visdrone" / "images" / "val"
    val_label_dir = project_root / "data" / "visdrone" / "labels" / "val"
    baseline_weights = project_root / "runs" / "detect" / "runs" / "baseline" / "yolov8n_visdrone" / "weights" / "best.pt"
    improved_weights = project_root / "runs" / "detect" / "runs" / "improved" / "yolov8s_visdrone" / "weights" / "best.pt"
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    print("Building COCO ground truth...")
    coco_gt_dict = build_coco_gt(val_image_dir, val_label_dir)
    img_files = sorted(f for f in val_image_dir.glob("*.jpg") if "_flip_temp" not in f.name)

    # Experiment 1: SAHI + TTA
    sahi_tta_result = run_sahi_tta(
        improved_weights, val_image_dir, val_label_dir,
        output_dir / "sahi_tta_metrics.json"
    )

    # Experiment 2: Threshold grid search
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(improved_weights),
        confidence_threshold=0.05, device="cuda:0",
    )
    threshold_result = run_threshold_search(
        sahi_model, val_image_dir, val_label_dir, img_files, coco_gt_dict,
        output_dir / "threshold_search.json"
    )

    # Experiment 3: Multi-model ensemble
    ensemble_result = run_ensemble(
        baseline_weights, improved_weights, val_image_dir, val_label_dir, coco_gt_dict,
        output_dir / "ensemble_metrics.json"
    )

    # Summary comparison
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)

    prev = {}
    for f in ["baseline_metrics.json", "improved_metrics.json", "sahi_metrics.json"]:
        p = output_dir / f
        if p.exists():
            with open(p) as fh:
                prev[f.replace("_metrics.json", "")] = json.load(fh)

    all_results = []
    for name, data in prev.items():
        all_results.append({"method": name, "mAP50": data.get("mAP50", 0), "mAP50_95": data.get("mAP50_95", 0)})
    all_results.append({"method": "SAHI+TTA", "mAP50": sahi_tta_result["mAP50"], "mAP50_95": sahi_tta_result["mAP50_95"]})
    all_results.append({"method": f"SAHI(conf={threshold_result['best_conf']})", "mAP50": threshold_result["best_mAP50"], "mAP50_95": threshold_result["best_mAP50_95"]})
    all_results.append({"method": "Ensemble", "mAP50": ensemble_result["mAP50"], "mAP50_95": ensemble_result["mAP50_95"]})

    all_results.sort(key=lambda x: x["mAP50"], reverse=True)

    print(f"\n  {'Method':<35} {'mAP@0.5':>10} {'mAP@0.5:0.95':>12}")
    print("  " + "-" * 60)
    for r in all_results:
        print(f"  {r['method']:<35} {r['mAP50']:>10.4f} {r['mAP50_95']:>12.4f}")

    summary = {"all_results": all_results, "best_method": all_results[0]["method"], "best_mAP50": all_results[0]["mAP50"]}
    with open(output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {output_dir / 'optimization_summary.json'}")


if __name__ == "__main__":
    main()
