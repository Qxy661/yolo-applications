# Usage Guide - Wind Turbine Blade Defect Detection

## Environment Setup

```bash
pip install ultralytics pillow matplotlib opencv-python
```

## Quick Start

### 1. Download Data

```bash
cd E:\yolo-wind-turbine\data
python scripts/download_datasets.py --all
```

If automatic download fails, manually download from:
- **Blade30**: Baidu Pan `https://pan.baidu.com/s/17kv5Xadz1QcSrvoG58WtBw` (code: 1234)
- **QQ datasets**: Check GitHub repos or Chinese AI platforms (AI Studio, ModelScope)

### 2. Clean Data

```bash
python scripts/clean_data.py --input raw/blade30 --output cleaned/blade30
python scripts/clean_data.py --input raw/qq767172261_6cls --output cleaned/qq6
python scripts/clean_data.py --input raw/qq767172261_uav5 --output cleaned/qq5
```

### 3. Convert Format

```bash
# Blade30 (JSON annotations)
python scripts/convert_format.py --input raw/blade30 --output processed/detection --format json

# QQ datasets (already YOLO format)
python scripts/convert_format.py --input raw/qq767172261_6cls --output processed/detection --format yolo
python scripts/convert_format.py --input raw/qq767172261_uav5 --output processed/detection --format yolo
```

### 4. Merge Datasets

```bash
python scripts/merge_datasets.py \
    --input raw/blade30 raw/qq767172261_6cls raw/qq767172261_uav5 \
    --output merged
```

### 5. Split Dataset

```bash
python scripts/split_dataset.py --input merged --output processed/detection
```

### 6. Run EDA

```bash
python scripts/eda_report.py --input processed/detection --output docs
```

## Training

### Basic Training

```python
from ultralytics import YOLO

# Load config
model = YOLO('yolo11n.pt')

# Train
results = model.train(
    data='data/wind_turbine.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='wind_turbine_v1',
)
```

### Recommended Hyperparameters

```yaml
# wind_turbine.yaml
path: data/processed/detection
train: images/train
val: images/val
test: images/test

names:
  0: crack
  1: erosion
  2: lightning
  3: peeling
  4: hole
```

### Advanced Training Options

```python
# With augmentation
model.train(
    data='data/wind_turbine.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    mosaic=1.0,           # Mosaic augmentation
    mixup=0.1,            # MixUp augmentation
    copy_paste=0.1,       # Copy-paste augmentation
    lr0=0.01,             # Initial learning rate
    lrf=0.01,             # Final learning rate factor
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    close_mosaic=15,      # Disable mosaic in last 15 epochs
)
```

### Inference

```python
from ultralytics import YOLO

model = YOLO('runs/detect/wind_turbine_v1/weights/best.pt')

# Single image
results = model('test_image.jpg')

# Directory
results = model('test_images/', save=True)

# Video
results = model('test_video.mp4', save=True)
```

### SAHI for Small Targets

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='runs/detect/wind_turbine_v1/weights/best.pt',
)

result = get_sliced_prediction(
    image='test_image.jpg',
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

## Data Augmentation Strategy

| Augmentation | mAP Gain | Priority |
|-------------|----------|----------|
| Mosaic (4-image) | +30% | ★★★★★ |
| MixUp | +10% | ★★★★★ |
| CopyPaste | +3.2% | ★★★★ |
| Resolution Progressive | High | ★★★★★ |

### Resolution Progressive Training

1. Phase 1: 640px for 50 epochs (fast convergence)
2. Phase 2: 1280px for 50 epochs (capture details)
3. Phase 3: Fine-tune on 1920px if needed

## Loss Function

Recommended: **EIoU + DFL**

- EIoU: Better gradient for aspect ratio (important for thin cracks)
- DFL: Distribution focal loss for precise boundary localization

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| mAP@0.5 | > 90% | Main metric |
| mAP@0.5:0.95 | > 70% | Strict metric |
| FPS | > 60 | Real-time capability |
| Model Size | < 10MB | Deployment friendly |

## Troubleshooting

### Common Issues

1. **Low mAP for small defects**: Use SAHI slicing or increase input resolution
2. **Class imbalance**: Use weighted loss or oversample minority classes
3. **Overfitting**: Add more augmentation, reduce model complexity
4. **Slow training**: Reduce batch size, use mixed precision (amp=True)

### GPU Memory Issues

```python
# Reduce batch size
model.train(batch=8)  # or 4

# Use gradient accumulation
model.train(batch=8, accumulation_steps=2)
```
