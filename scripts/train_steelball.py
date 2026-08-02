"""
Steel-ball detection training (contest H-track) — closed-loop framework.

Reuses the same YOLO closed-loop paradigm as VisDrone:
    data → train(finetune) → evaluate → deploy

Data: /root/yolo-m2/steelball_data (1943 imgs, train 1555 + val 388, single class)

Usage:
    python scripts/train_steelball.py --model yolo26n.pt --imgsz 640 --epochs 50
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_YAML = "/root/yolo-m2/steelball_data/steelball.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="/mnt/e/yolo-visdrone/runs/steelball")
    parser.add_argument("--name", default="steelball_yolo26")
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"模型: {args.model}, imgsz: {args.imgsz}, epochs: {args.epochs}")
    model = YOLO(args.model)

    model.train(
        data=DATA_YAML,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        mosaic=1.0,
        copy_paste=0.3,
        scale=0.5,
        degrees=90,        # 钢珠方向任意
        close_mosaic=10,
        patience=30,
        device=0,
    )

    # Evaluate
    print("\n=== 评估 ===")
    model.val(data=DATA_YAML)

    print(f"\n完成! 结果在 {args.project}/{args.name}")
    print("下一步: 导出 ONNX/TensorRT, 实时推理")


if __name__ == "__main__":
    main()
