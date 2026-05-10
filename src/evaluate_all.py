"""
标准化全量实验评估脚本
在 val 集上统一评估所有模型，生成规范化对比报告
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import PIL.Image
_pil_open = PIL.Image.open

sys.path.insert(0, str(Path(__file__).parent))
import register_custom_modules  # noqa: F401

from ultralytics import YOLO

PIL.Image.open = _pil_open

PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML = str(PROJECT_ROOT / "data" / "visdrone" / "visdrone.yaml")
RESULTS_DIR = PROJECT_ROOT / "results"

# ── 模型定义 ──────────────────────────────────────────────
MODELS = {
    "baseline": {
        "name": "YOLOv8n Baseline",
        "weights": PROJECT_ROOT / "runs" / "baseline" / "yolov8n_visdrone" / "weights" / "best.pt",
        "imgsz": 640,
        "description": "YOLOv8n, 640px, 默认增强, 50 epochs",
    },
    "improved": {
        "name": "YOLOv8s Improved",
        "weights": PROJECT_ROOT / "runs" / "improved" / "yolov8s_visdrone" / "weights" / "best.pt",
        "imgsz": 800,
        "description": "YOLOv8s, 800px, 强增强 (MixUp+CopyPaste+Erasing), 80 epochs",
    },
    "p2_cbam": {
        "name": "YOLOv8s P2+CBAM",
        "weights": PROJECT_ROOT / "runs" / "p2" / "yolov8s_p2_cbam" / "weights" / "best.pt",
        "imgsz": 800,
        "description": "YOLOv8s + P2检测头(160x160) + CBAM注意力, 800px, 80 epochs",
    },
}


def evaluate_model(key, cfg):
    """评估单个模型，返回标准化指标"""
    weights = cfg["weights"]
    if not weights.exists():
        print(f"  [SKIP] {cfg['name']}: 权重文件不存在 {weights}")
        return None

    print(f"\n{'='*60}")
    print(f"  评估: {cfg['name']}")
    print(f"  权重: {weights}")
    print(f"  输入: {cfg['imgsz']}px")
    print(f"{'='*60}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=DATA_YAML,
        imgsz=cfg["imgsz"],
        batch=16,
        conf=0.001,
        iou=0.6,
        device="0",
        verbose=True,
        plots=False,
    )

    # 提取标准化指标
    result = {
        "model": cfg["name"],
        "key": key,
        "description": cfg["description"],
        "weights": str(weights),
        "imgsz": cfg["imgsz"],
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class_ap50": {},
    }

    # 各类别 AP
    names = metrics.names
    for i, name in names.items():
        result["per_class_ap50"][name] = round(float(metrics.box.ap50[i]), 4)

    # 打印结果
    print(f"\n  mAP@0.5:     {result['metrics']['mAP50']}")
    print(f"  mAP@0.5:0.95: {result['metrics']['mAP50_95']}")
    print(f"  Precision:   {result['metrics']['precision']}")
    print(f"  Recall:      {result['metrics']['recall']}")
    print(f"\n  各类别 AP@0.5:")
    for name, ap in result["per_class_ap50"].items():
        print(f"    {name:20s}: {ap:.4f}")

    return result


def generate_report(all_results):
    """生成标准化对比报告"""
    if not all_results:
        print("[WARN] 无评估结果")
        return

    # 保存 JSON
    report = {
        "evaluation_time": datetime.now().isoformat(),
        "dataset": "VisDrone2019-DET val (548 images)",
        "models_evaluated": len(all_results),
        "results": all_results,
    }

    json_path = RESULTS_DIR / "standardized_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存: {json_path}")

    # 生成 Markdown 对比表
    md_lines = [
        "# 标准化实验对比报告",
        "",
        f"> 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 数据集: VisDrone2019-DET val (548 images)",
        f"> 评估设备: RTX 4060 Laptop",
        "",
        "---",
        "",
        "## 整体指标对比",
        "",
        "| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | vs Baseline |",
        "|------|---------|--------------|-----------|--------|-------------|",
    ]

    baseline_map50 = None
    for r in all_results:
        m = r["metrics"]
        if baseline_map50 is None:
            baseline_map50 = m["mAP50"]
        delta = ((m["mAP50"] - baseline_map50) / baseline_map50 * 100) if baseline_map50 else 0
        delta_str = f"+{delta:.1f}%" if delta > 0 else ("-" if baseline_map50 is None else f"{delta:.1f}%")
        md_lines.append(
            f"| {r['model']} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {delta_str} |"
        )

    # 各类别 AP 对比
    if len(all_results) >= 2:
        md_lines += [
            "",
            "## 各类别 AP@0.5 对比",
            "",
        ]
        header = "| 类别 | " + " | ".join(r["model"] for r in all_results) + " |"
        sep = "|------|" + "|".join("------" for _ in all_results) + "|"
        md_lines += [header, sep]

        classes = list(all_results[0]["per_class_ap50"].keys())
        for cls in classes:
            vals = [f"{r['per_class_ap50'].get(cls, 0):.4f}" for r in all_results]
            md_lines.append(f"| {cls} | " + " | ".join(vals) + " |")

    md_lines += [
        "",
        "## 模型配置",
        "",
        "| 模型 | 输入尺寸 | 参数量 | 检测点 | 特殊配置 |",
        "|------|---------|--------|--------|---------|",
    ]
    model_configs = {
        "YOLOv8n Baseline": ("640", "3.2M", "8400", "默认增强"),
        "YOLOv8s Improved": ("800", "11.2M", "8400", "MixUp+CopyPaste+Erasing"),
        "YOLOv8s P2+CBAM": ("800", "14.5M", "34000", "P2检测头+CBAM注意力"),
    }
    for r in all_results:
        cfg = model_configs.get(r["model"], ("?", "?", "?", "?"))
        md_lines.append(f"| {r['model']} | {cfg[0]} | {cfg[1]} | {cfg[2]} | {cfg[3]} |")

    md_path = RESULTS_DIR / "standardized_eval.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"对比报告已保存: {md_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for key, cfg in MODELS.items():
        result = evaluate_model(key, cfg)
        if result:
            all_results.append(result)

    generate_report(all_results)

    # 打印最终排名
    if all_results:
        print(f"\n{'='*60}")
        print("  最终排名 (按 mAP@0.5)")
        print(f"{'='*60}")
        ranked = sorted(all_results, key=lambda r: r["metrics"]["mAP50"], reverse=True)
        baseline = all_results[0]["metrics"]["mAP50"]
        for i, r in enumerate(ranked, 1):
            m = r["metrics"]
            delta = (m["mAP50"] - baseline) / baseline * 100
            print(f"  {i}. {r['model']:25s}  mAP={m['mAP50']:.4f}  (+{delta:.1f}%)")


if __name__ == "__main__":
    main()
