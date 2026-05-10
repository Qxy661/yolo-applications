"""
改进模型训练脚本
方案: YOLOv8s + 更大输入尺寸 + 更强数据增强
"""
from ultralytics import YOLO
from pathlib import Path


def train_improved():
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / "data" / "visdrone" / "visdrone.yaml"

    # 使用 YOLOv8s (比 n 大一号，精度更高)
    model = YOLO("yolov8s.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=800,         # 更大输入尺寸，保留更多小目标信息
        batch=8,           # 更大图片需要更小 batch
        workers=4,
        device="0",
        project=str(project_root / "runs" / "improved"),
        name="yolov8s_visdrone_improved",
        exist_ok=True,
        pretrained=True,
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
        # 更强的数据增强
        hsv_h=0.02,        # 色调增强
        hsv_s=0.8,         # 饱和度增强
        hsv_v=0.5,         # 亮度增强
        degrees=10.0,      # 旋转增强
        translate=0.15,    # 平移增强
        scale=0.6,         # 缩放增强
        shear=5.0,         # 剪切增强
        perspective=0.001, # 透视增强
        fliplr=0.5,        # 水平翻转
        mosaic=1.0,        # 马赛克增强
        mixup=0.15,        # MixUp 增强
        copy_paste=0.1,    # 复制粘贴增强
        erasing=0.5,       # 随机擦除
    )

    print(f"\n改进模型训练完成!")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    return results


if __name__ == "__main__":
    train_improved()
