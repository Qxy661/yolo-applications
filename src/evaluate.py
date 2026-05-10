"""
YOLOv8 评估脚本 — 计算 mAP 和小目标精度
"""
import argparse
import json
from ultralytics import YOLO
from pathlib import Path


def evaluate(args):
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / "data" / "visdrone" / "visdrone.yaml"

    # 加载模型
    model = YOLO(args.weights)

    # 在验证集上评估
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_root / "runs" / "eval"),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # 打印结果
    print("\n" + "=" * 50)
    print("  评估结果")
    print("=" * 50)
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")

    # 每类 mAP
    print("\n  各类别 mAP@0.5:")
    names = metrics.names
    for i, ap in enumerate(metrics.box.ap50):
        print(f"    {names[i]:20s}: {ap:.4f}")

    # 保存结果
    results_file = project_root / "runs" / "eval" / args.name / "metrics.json"
    results = {
        "mAP50": metrics.box.map50,
        "mAP50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "per_class_ap50": {names[i]: float(ap) for i, ap in enumerate(metrics.box.ap50)},
    }
    results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  结果已保存: {results_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Evaluation")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批量大小")
    parser.add_argument("--device", type=str, default="0", help="GPU设备号")
    parser.add_argument("--name", type=str, default="yolov8n_eval", help="实验名称")
    args = parser.parse_args()
    evaluate(args)
