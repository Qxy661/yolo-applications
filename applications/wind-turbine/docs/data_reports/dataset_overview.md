# Wind Turbine Blade Defect Detection Dataset

## Overview

This dataset is designed for wind turbine blade defect detection using YOLO-based object detection models. It combines multiple public datasets to provide comprehensive coverage of blade defects.

## Defect Classes (5 classes)

| ID | Class | Chinese | Description |
|----|-------|---------|-------------|
| 0 | crack | 裂纹 | Surface cracks on blade |
| 1 | erosion | 侵蚀 | Surface erosion, dirt, oil leakage |
| 2 | lightning | 雷击 | Lightning strike damage |
| 3 | peeling | 涂层脱落 | Paint/coating peeling |
| 4 | hole | 孔洞 | Pin holes or larger holes |

## Data Sources

| Dataset | Source | Images | Format | Classes |
|---------|--------|--------|--------|---------|
| Blade30 | Renewable Energy 2023 | 1,302 | JSON | crack, erosion, lightning, peeling, hole |
| QQ767172261 6-class | GitHub | 3,282 | YOLO txt | crack, erosion, dirt, oil_leakage, pu_tape, pin_hole |
| QQ767172261 UAV 5-class | GitHub | 4,467 | YOLO txt | oil_leakage, dirt, paint, le_erosion, pu_tape |
| WTBD | Nature Scientific Data 2026 | TBD | TBD | TBD |

## Directory Structure

```
data/
├── raw/                          # Original downloaded data
│   ├── blade30/                  # Blade30 dataset
│   ├── qq767172261_6cls/         # QQ 6-class dataset
│   ├── qq767172261_uav5/         # QQ UAV 5-class dataset
│   └── wtbd/                     # WTBD dataset
├── processed/                    # Processed data (YOLO format)
│   ├── detection/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── segmentation/
│       ├── images/
│       └── masks/
├── scripts/                      # Processing scripts
│   ├── download_datasets.py
│   ├── clean_data.py
│   ├── convert_format.py
│   ├── merge_datasets.py
│   ├── split_dataset.py
│   └── eda_report.py
├── docs/                         # Documentation
│   ├── dataset_overview.md
│   ├── eda_report.md
│   ├── quality_assessment.md
│   └── usage_guide.md
└── wind_turbine.yaml             # YOLO config
```

## Label Format

YOLO format: `class_id center_x center_y width height` (normalized 0-1)

```
0 0.512345 0.234567 0.051234 0.023456
1 0.345678 0.678901 0.123456 0.067890
```

## Usage

### Quick Start

```bash
# 1. Download datasets
python scripts/download_datasets.py --all

# 2. Clean data
python scripts/clean_data.py --input raw/blade30 --output cleaned/blade30

# 3. Convert format
python scripts/convert_format.py --input raw/blade30 --output processed/detection

# 4. Merge datasets
python scripts/merge_datasets.py --input raw/blade30 raw/qq767172261_6cls raw/qq767172261_uav5 --output merged

# 5. Split dataset
python scripts/split_dataset.py --input merged --output processed/detection

# 6. EDA analysis
python scripts/eda_report.py --input processed/detection --output docs
```

### Training with YOLOv11

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # Load pretrained model
results = model.train(
    data='data/wind_turbine.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
)
```

## Citation

If you use this dataset, please cite the original sources:
- Blade30: Yang et al., "Blade30: A dataset for wind turbine blade inspection", Renewable Energy, 2023
- QQ767172261 datasets: Available on GitHub
- WTBD: Available in Nature Scientific Data 2026
