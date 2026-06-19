# 模型权重

本目录包含在 VisDrone2019-DET 上训练好的 YOLO 模型权重。

## 训练好的权重

| 文件 | 大小 | 模型 | 输入尺寸 | mAP@0.5 | 说明 |
|:----|:---:|:----:|:--------:|:-------:|:----:|
| [**best.pt**](best.pt) | 22MB | YOLOv8s | 800px | **0.4903** 🏆 | **最佳模型**: Improved + SAHI(conf=0.05) |
| [baseline.pt](baseline.pt) | 6MB | YOLOv8n | 640px | 0.2979 | 基线对照 |

## 如何使用

```python
from ultralytics import YOLO

# 加载最佳模型
model = YOLO("weights/best.pt")
results = model("your_image.jpg")

# 或使用 SAHI 达到最佳效果 (mAP 0.4903)
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="weights/best.pt",
    confidence_threshold=0.05,
    device="cuda:0",
)
```

## 预训练权重

预训练 COCO 权重（`yolov8n.pt`, `yolov8s.pt` 等）放在 `pretrained/` 目录下。
训练时会自动下载，不随仓库上传。

## 权重来源

| 权重 | 来源 | 训练脚本 |
|:----|:----|:--------|
| best.pt | `runs/detect/runs/improved/yolov8s_visdrone/weights/best.pt` | `src/train_improved.py` |
| baseline.pt | `runs/detect/runs/baseline/yolov8n_visdrone/weights/best.pt` | `src/train.py` |

> P2+CBAM 权重（112MB）训练尚未收敛，暂不发布。
