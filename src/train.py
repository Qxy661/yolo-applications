"""
YOLOv8 训练脚本 — VisDrone 低空小目标检测
"""
import argparse
from ultralytics import YOLO
from pathlib import Path


def train(args):
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / "data" / "visdrone" / "visdrone.yaml"

    # 加载模型
    if args.weights:
        model = YOLO(args.weights)
    else:
        model = YOLO(f"{args.model}.pt")

    # 训练参数
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project_root / "runs" / args.name),
        name=args.exp,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        amp=True,
        val=True,
        save=True,
        save_period=-1,
        plots=True,
        verbose=True,
    )

    print(f"\n训练完成！结果保存在: {project_root / 'runs' / args.name / args.exp}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 VisDrone Training")
    parser.add_argument("--model", type=str, default="yolov8n", help="模型: yolov8n/s/m/l/x")
    parser.add_argument("--weights", type=str, default=None, help="自定义权重路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批量大小")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--device", type=str, default="0", help="GPU设备号")
    parser.add_argument("--name", type=str, default="baseline", help="实验类别")
    parser.add_argument("--exp", type=str, default="yolov8n_visdrone", help="实验名称")
    args = parser.parse_args()
    train(args)
