"""
SAHI 检测可视化脚本
生成 SAHI vs 普通推理的并排对比图
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


def draw_sahi_boxes(img, result, class_names, conf_threshold=0.25):
    """在图片上绘制 SAHI 检测框"""
    colors = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187)
    ]

    for pred in result.object_prediction_list:
        score = pred.score.value
        if score < conf_threshold:
            continue

        bbox = pred.bbox.to_voc_bbox()  # [x1, y1, x2, y2]
        cls_id = pred.category.id
        cls_name = pred.category.name if pred.category.name else str(cls_id)
        color = colors[cls_id % len(colors)]

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img


def run_sahi_detect(image_path, weights_path, output_path, conf=0.25, slice_size=640, overlap=0.2):
    """对单张图片运行 SAHI 检测"""
    # SAHI 模型
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=conf,
        device="cuda:0",
    )

    # 普通 YOLO 模型
    yolo_model = YOLO(str(weights_path))

    img = cv2.imread(str(image_path))

    # 普通推理
    yolo_results = yolo_model.predict(source=str(image_path), conf=conf, verbose=False)
    yolo_img = yolo_results[0].plot(conf=True, labels=True, boxes=True, line_width=2, font_size=16)

    # SAHI 推理
    sahi_result = get_sliced_prediction(
        image=str(image_path),
        detection_model=sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=True,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
        verbose=0,
    )

    sahi_img = img.copy()
    draw_sahi_boxes(sahi_img, sahi_result, None, conf)

    # 添加标签
    cv2.putText(yolo_img, "YOLOv8s (normal)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(sahi_img, "SAHI (sliced)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 统计检测数
    yolo_count = len(yolo_results[0].boxes)
    sahi_count = len([p for p in sahi_result.object_prediction_list if p.score.value >= conf])
    cv2.putText(yolo_img, f"Detections: {yolo_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(sahi_img, f"Detections: {sahi_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 水平拼接
    h = min(yolo_img.shape[0], sahi_img.shape[0])
    yolo_resized = cv2.resize(yolo_img, (int(yolo_img.shape[1] * h / yolo_img.shape[0]), h))
    sahi_resized = cv2.resize(sahi_img, (int(sahi_img.shape[1] * h / sahi_img.shape[0]), h))
    combined = np.hstack([yolo_resized, sahi_resized])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    print(f"  {Path(image_path).stem}: YOLO={yolo_count}, SAHI={sahi_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHI Detection Visualization")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--source", type=str, default=None, help="单张图片或目录")
    parser.add_argument("--output", type=str, default="results/sahi_compare")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    weights = Path(args.weights) if args.weights else project_root / "runs" / "detect" / "runs" / "improved" / "yolov8s_visdrone" / "weights" / "best.pt"
    source = Path(args.source) if args.source else project_root / "data" / "visdrone" / "images" / "val"
    output_dir = project_root / args.output

    img_files = sorted(source.glob("*.jpg"))[:args.max]
    print(f"生成 {len(img_files)} 张 SAHI 对比图...")

    for i, img_file in enumerate(img_files):
        out_path = output_dir / f"sahi_compare_{img_file.stem}.jpg"
        run_sahi_detect(img_file, weights, out_path, args.conf)

    print(f"\n对比图已保存: {output_dir}")
