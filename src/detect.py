"""
YOLOv8 推理/检测脚本
"""
import argparse
from ultralytics import YOLO
from pathlib import Path


def detect(args):
    project_root = Path(__file__).parent.parent

    # 加载模型
    model = YOLO(args.weights)

    # 推理
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(project_root / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    print(f"\n检测完成！结果保存在: {project_root / 'runs' / 'detect' / args.name}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Detection")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="输入源 (图片/视频/目录)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--device", type=str, default="0", help="GPU设备号")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")
    args = parser.parse_args()
    detect(args)
