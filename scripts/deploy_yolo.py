"""
YOLO 部署脚本 — 导出 + 实时推理（闭环的最后一步）

Steps:
  1. Export best.pt -> ONNX (cross-platform)
  2. Export best.pt -> TensorRT engine (GPU-accelerated, optional)
  3. Real-time inference (webcam / video / image)

Usage:
    python scripts/deploy_yolo.py export --model runs/visdrone/best.pt --imgsz 800
    python scripts/deploy_yolo.py infer --model runs/visdrone/best.onnx --source 0
    python scripts/deploy_yolo.py infer --model runs/visdrone/best.engine --source video.mp4
"""
import argparse
import os
import sys


def export_model(model_path, imgsz, formats=("onnx",), half=False):
    """Export model to ONNX / TensorRT."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    for fmt in formats:
        print(f"=== 导出 {fmt} ===")
        out = model.export(
            format=fmt,
            imgsz=imgsz,
            half=half if fmt == "engine" else False,
            simplify=(fmt == "onnx"),
            device=0 if fmt == "engine" else None,
        )
        print(f"已导出: {out}")


def infer(model_path, source, conf=0.25, imgsz=640):
    """Real-time / batch inference."""
    from ultralytics import YOLO
    model = YOLO(model_path)

    if source == "0" or str(source).isdigit():
        # Webcam real-time
        print(f"=== 摄像头实时推理 (conf={conf}) ===")
        results = model.predict(source=int(source), stream=True,
                                conf=conf, imgsz=imgsz)
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf_v = box.conf[0].item()
                    cls = int(box.cls[0])
                    print(f"  检测到: class={cls}, conf={conf_v:.3f}, "
                          f"box={[round(x) for x in xyxy]}")
    else:
        # Image / video
        print(f"=== 推理: {source} ===")
        results = model.predict(source=source, conf=conf, imgsz=imgsz)
        for i, r in enumerate(results):
            print(f"  {r.path}: {len(r.boxes) if r.boxes is not None else 0} 个目标")
            r.save(filename=f"output_{i}.jpg")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # export subcommand
    pe = sub.add_parser("export", help="导出 ONNX/TensorRT")
    pe.add_argument("--model", required=True)
    pe.add_argument("--imgsz", type=int, default=640)
    pe.add_argument("--format", nargs="+", default=["onnx"],
                    choices=["onnx", "engine"])
    pe.add_argument("--half", action="store_true", help="FP16 for engine")

    # infer subcommand
    pi = sub.add_parser("infer", help="推理 (webcam/video/image)")
    pi.add_argument("--model", required=True)
    pi.add_argument("--source", required=True, help="0=webcam, path=video/img")
    pi.add_argument("--conf", type=float, default=0.25)
    pi.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args()

    if args.cmd == "export":
        export_model(args.model, args.imgsz, args.format, args.half)
    elif args.cmd == "infer":
        infer(args.model, args.source, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
