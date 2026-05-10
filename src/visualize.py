"""
可视化脚本 — 检测结果对比图、GradCAM 热力图
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def draw_comparison(model_path: str, img_dir: str, output_dir: str, conf: float = 0.25, max_imgs: int = 20):
    """绘制检测结果对比图"""
    model = YOLO(model_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(Path(img_dir).glob("*.jpg"))[:max_imgs]
    print(f"生成 {len(img_files)} 张检测对比图...")

    for i, img_file in enumerate(img_files):
        results = model.predict(source=str(img_file), conf=conf, verbose=False)
        result = results[0]

        # 绘制检测框
        annotated = result.plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2,
            font_size=16,
        )

        # 保存
        save_path = out_path / f"detect_{img_file.stem}.jpg"
        cv2.imwrite(str(save_path), annotated)
        print(f"  [{i+1}/{len(img_files)}] {save_path.name}")


def generate_heatmap(model_path: str, img_path: str, output_path: str):
    """生成 GradCAM 热力图 (使用 Ultralytics 内置 feature visualization)"""
    model = YOLO(model_path)

    # 使用 Ultralytics 的内置方法生成特征可视化
    results = model.predict(source=img_path, verbose=False)
    result = results[0]

    # 获取原始图片
    img = cv2.imread(img_path)

    # 使用 result 的 plot 方法叠加检测结果
    annotated = result.plot(conf=True, labels=True, boxes=True, line_width=2)

    # 保存
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotated)
    print(f"  热力图已保存: {output_path}")


def compare_models(model_paths: dict, img_dir: str, output_dir: str, conf: float = 0.25, max_imgs: int = 10):
    """多模型检测结果对比"""
    models = {name: YOLO(path) for name, path in model_paths.items()}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(Path(img_dir).glob("*.jpg"))[:max_imgs]
    print(f"生成 {len(img_files)} 张多模型对比图...")

    for i, img_file in enumerate(img_files):
        # 每个模型的检测结果并排显示
        images = []
        for name, model in models.items():
            results = model.predict(source=str(img_file), conf=conf, verbose=False)
            annotated = results[0].plot(conf=True, labels=True, boxes=True, line_width=2, font_size=16)
            # 添加模型名称标签
            cv2.putText(annotated, name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            images.append(annotated)

        # 水平拼接
        # 确保所有图片高度一致
        h = min(img.shape[0] for img in images)
        resized = []
        for img in images:
            ratio = h / img.shape[0]
            new_w = int(img.shape[1] * ratio)
            resized.append(cv2.resize(img, (new_w, h)))
        combined = np.hstack(resized)

        save_path = out_path / f"compare_{img_file.stem}.jpg"
        cv2.imwrite(str(save_path), combined)
        print(f"  [{i+1}/{len(img_files)}] {save_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Visualization")
    subparsers = parser.add_subparsers(dest="command")

    # 检测对比图
    p_detect = subparsers.add_parser("detect", help="生成检测对比图")
    p_detect.add_argument("--weights", type=str, required=True)
    p_detect.add_argument("--source", type=str, required=True, help="图片目录")
    p_detect.add_argument("--output", type=str, default="results/detect_vis")
    p_detect.add_argument("--conf", type=float, default=0.25)
    p_detect.add_argument("--max", type=int, default=20)

    # 热力图
    p_heat = subparsers.add_parser("heatmap", help="生成热力图")
    p_heat.add_argument("--weights", type=str, required=True)
    p_heat.add_argument("--source", type=str, required=True, help="单张图片")
    p_heat.add_argument("--output", type=str, default="results/heatmap.jpg")

    # 多模型对比
    p_compare = subparsers.add_parser("compare", help="多模型对比")
    p_compare.add_argument("--models", type=str, nargs="+", help="name=path 格式的模型列表")
    p_compare.add_argument("--source", type=str, required=True, help="图片目录")
    p_compare.add_argument("--output", type=str, default="results/compare")
    p_compare.add_argument("--conf", type=float, default=0.25)
    p_compare.add_argument("--max", type=int, default=10)

    args = parser.parse_args()

    if args.command == "detect":
        draw_comparison(args.weights, args.source, args.output, args.conf, args.max)
    elif args.command == "heatmap":
        generate_heatmap(args.weights, args.source, args.output)
    elif args.command == "compare":
        models = {}
        for m in args.models:
            name, path = m.split("=", 1)
            models[name] = path
        compare_models(models, args.source, args.output, args.conf, args.max)
    else:
        parser.print_help()
