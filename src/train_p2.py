"""
YOLOv8s + P2 小目标检测头 + CBAM 训练脚本
在标准 YOLOv8s 基础上:
  - 增加 P2 (160×160) 检测层，专门检测 <16px 的小目标
  - 在 Backbone 的 P2/P3/P4 后插入 CBAM 注意力模块
  - 检测点从 8400 增加到 34000 (P2:25600 + P3:6400 + P4:1600 + P5:400)
"""
import sys
from pathlib import Path

import PIL.Image
_pil_open = PIL.Image.open  # pi_heif workaround

sys.path.insert(0, str(Path(__file__).parent))
import register_custom_modules  # noqa: F401 — 注册 CBAM 到 ultralytics

from ultralytics import YOLO

PIL.Image.open = _pil_open


def main():
    project_root = Path(__file__).parent.parent
    data_yaml = str(project_root / "data" / "visdrone" / "visdrone.yaml")
    model_yaml = str(project_root / "configs" / "yolov8s-p2.yaml")
    weights_path = project_root / "yolov8s.pt"

    print("=" * 50)
    print("  YOLOv8s + P2 小目标检测头 + CBAM")
    print("=" * 50)

    # 1. 从 yaml 创建自定义模型
    model = YOLO(model_yaml)

    # 2. 加载 COCO 预训练权重 (按形状匹配，CBAM/P2 层随机初始化)
    if weights_path.exists():
        register_custom_modules.load_compatible_weights(model.model, str(weights_path))
    else:
        print(f"[WARN] 预训练权重 {weights_path} 不存在，从头训练")

    # 3. 训练
    results = model.train(
        data=data_yaml,
        epochs=80,
        imgsz=640,
        batch=4,
        workers=4,
        device="0",
        project=str(project_root / "runs" / "p2"),
        name="yolov8s_p2_cbam",
        exist_ok=True,
        pretrained=False,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=15,
        amp=True,
        val=True,
        save=True,
        plots=True,
        verbose=True,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.15,
        scale=0.6,
        shear=5.0,
        perspective=0.001,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        erasing=0.5,
    )

    print(f"\n训练完成！")
    print(f"mAP@0.5:    {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"权重保存在: runs/p2/yolov8s_p2_cbam/weights/")


if __name__ == "__main__":
    main()
