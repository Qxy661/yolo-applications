"""
SAHI 切片推理评估脚本
使用 SAHI 对 VisDrone 验证集进行切片推理，计算 mAP
"""
import json
import time
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm


# VisDrone 类别映射 (YOLO class_id -> COCO category_id)
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def build_coco_gt(val_image_dir, val_label_dir):
    """从 YOLO 标注构建 COCO 格式的 ground truth"""
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 类别
    for i, name in enumerate(VISDRONE_CLASSES):
        coco["categories"].append({"id": i, "name": name})

    ann_id = 0
    img_files = sorted(Path(val_image_dir).glob("*.jpg"))

    for img_id, img_file in enumerate(img_files):
        img = Image.open(img_file)
        w, h = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": w,
            "height": h
        })

        # 读取对应标注
        label_file = Path(val_label_dir) / (img_file.stem + ".txt")
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # YOLO xywh (normalized) -> COCO xywh (pixel)
                    x = (x_c - bw / 2) * w
                    y = (y_c - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [x, y, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0
                    })
                    ann_id += 1

    return coco


def run_sahi_evaluation():
    """运行 SAHI 切片推理评估"""
    project_root = Path(__file__).parent.parent
    val_image_dir = project_root / "data" / "visdrone" / "images" / "val"
    val_label_dir = project_root / "data" / "visdrone" / "labels" / "val"
    weights_path = project_root / "runs" / "detect" / "runs" / "improved" / "yolov8s_visdrone" / "weights" / "best.pt"

    print("=" * 60)
    print("  SAHI 切片推理评估")
    print("=" * 60)

    # 1. 构建 COCO ground truth
    print("\n[1/4] 构建 COCO ground truth...")
    coco_gt_dict = build_coco_gt(val_image_dir, val_label_dir)
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    print(f"  图片数: {len(coco_gt_dict['images'])}")
    print(f"  标注数: {len(coco_gt_dict['annotations'])}")

    # 2. 加载模型
    print("\n[2/4] 加载 SAHI 模型...")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=0.25,
        device="cuda:0",
    )
    print(f"  模型: {weights_path.name}")

    # 3. SAHI 切片推理
    print("\n[3/4] SAHI 切片推理...")
    coco_results = []
    img_files = sorted(val_image_dir.glob("*.jpg"))

    start_time = time.time()
    for img_id, img_file in enumerate(tqdm(img_files, desc="推理中")):
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

        # 转换为 COCO 格式
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_coco_bbox()  # [x, y, w, h]
            coco_results.append({
                "image_id": img_id,
                "category_id": pred.category.id,
                "bbox": bbox,
                "score": pred.score.value,
            })

    elapsed = time.time() - start_time
    print(f"  耗时: {elapsed:.1f}s ({elapsed/len(img_files):.2f}s/张)")
    print(f"  预测数: {len(coco_results)}")

    # 4. COCO 评估
    print("\n[4/4] COCO mAP 评估...")
    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 提取指标
    stats = coco_eval.stats
    results = {
        "method": "SAHI + YOLOv8s",
        "slice_size": [640, 640],
        "overlap_ratio": 0.2,
        "mAP50": float(stats[1]),
        "mAP50_95": float(stats[0]),
        "precision": float(stats[2]) if len(stats) > 2 else 0.0,
        "total_predictions": len(coco_results),
        "inference_time_s": round(elapsed, 1),
        "time_per_image_s": round(elapsed / len(img_files), 2),
    }

    # 计算各类别 AP (从已有 precision 数组提取)
    # prec shape: [T, R, K, A, M] - T=IoU(10), R=recall(101), K=class(10), A=area(4), M=maxDet(3)
    prec = coco_eval.eval["precision"]
    per_class_ap = {}
    for i, name in enumerate(VISDRONE_CLASSES):
        # AP@0.5 (IoU index 0) for this class, all areas, maxDet=100
        ap50_slice = prec[0, :, i, 0, 2]  # IoU=0.5, all recall, class i, area=all, maxDet=100
        valid = ap50_slice[ap50_slice > -1]
        ap50 = float(valid.mean()) if len(valid) > 0 else 0.0
        per_class_ap[name] = round(ap50, 4)

    results["per_class_ap50"] = per_class_ap

    # 打印结果
    print("\n" + "=" * 60)
    print("  SAHI 评估结果")
    print("=" * 60)
    print(f"  mAP@0.5:    {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP50_95']:.4f}")
    print(f"\n  各类别 AP@0.5:")
    for name, ap in per_class_ap.items():
        print(f"    {name:20s}: {ap:.4f}")

    # 保存结果
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "sahi_metrics.json"
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  结果已保存: {output_file}")

    return results


if __name__ == "__main__":
    run_sahi_evaluation()
