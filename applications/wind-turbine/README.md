# 风电场叶片缺陷检测 — Wind Turbine Blade Defect Detection

基于 YOLOv11 的风力发电机叶片表面缺陷检测系统。

## 项目简介

本项目使用 YOLOv11 目标检测模型，实现对风力发电机叶片表面缺陷的自动识别，支持 **5 类核心缺陷**：

| 类别ID | 中文名称 | 英文名称 | 说明 |
|--------|---------|---------|------|
| 0 | 叶片裂纹 | Crack | 表面裂纹、疲劳裂纹 |
| 1 | 叶片破损 | Breakage | 缺损、断裂、孔洞 |
| 2 | 雷击损伤 | Lightning | 雷击烧蚀、碳化痕迹 |
| 3 | 涂层脱落 | Peeling | 涂层剥落、脱漆 |
| 4 | 边缘侵蚀 | Erosion | 前缘侵蚀、边缘磨损 |

## 技术路线

```
文献调研 → 数据集构建 → YOLOv11基线训练 → 改进优化 → 部署评估
   ↓           ↓              ↓               ↓            ↓
 论文精读   多源数据合并    ultralytics      注意力机制    ONNX导出
 5类缺陷   YOLO格式统一    预训练权重        数据增强      推理优化
```

## 项目结构

```
yolo-wind-turbine/
├── README.md                    # 项目说明
├── LICENSE                      # MIT License
├── PROJECT_PLAN.md              # 详细项目计划
├── TECHNICAL_REPORT.md          # 技术报告(文献综述+数据集调研+方案设计)
├── requirements.txt             # Python 依赖
├── setup_env.bat                # 一键环境配置
├── data/
│   ├── wind_turbine.yaml        # 数据集配置
│   ├── images/{train,val,test}/ # 图片目录
│   ├── labels/{train,val,test}/ # YOLO标注
│   └── raw/                     # 原始数据集
├── src/
│   ├── train.py                 # 基线训练脚本
│   ├── train_improved.py        # 改进训练(CA/ECA/多尺度)
│   ├── detect.py                # 推理脚本
│   ├── evaluate.py              # 评估脚本
│   └── utils.py                 # 工具函数
├── configs/
│   ├── yolov11n.yaml            # YOLOv11-nano基线配置
│   └── yolov11s.yaml            # YOLOv11-small+CA注意力配置
├── scripts/
│   ├── download_datasets.py     # 数据集下载脚本
│   ├── merge_datasets.py        # 多源数据合并
│   └── split_dataset.py         # 数据集划分(8:1:1)
├── docs/
│   ├── literature_review.md     # 文献综述
│   └── experiment_log.md        # 实验记录
├── results/                     # 实验结果(JSON)
└── runs/                        # 训练权重
```

## 快速开始

### 环境配置

```bash
# 激活 conda 环境
conda activate yolo-project

# 安装依赖
pip install -r requirements.txt

# 验证环境
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

### 数据集准备

```bash
# 1. 下载数据集(需手动从百度网盘/Google Drive下载)
# 2. 放入 data/raw/ 目录
# 3. 合并并划分
python scripts/merge_datasets.py
python scripts/split_dataset.py
```

### 训练

```bash
# 基线训练
python src/train.py --model yolo11n.pt --epochs 100 --batch 16

# 改进训练(带CA注意力)
python src/train_improved.py --model yolo11n.pt --use-ca --epochs 100
```

### 评估与推理

```bash
# 评估
python src/evaluate.py --weights runs/train/baseline/weights/best.pt

# 推理
python src/detect.py --weights runs/train/baseline/weights/best.pt --source test_images/
```

## 数据集

本项目需要多源数据合并，详见 `TECHNICAL_REPORT.md` 第3章。

| 数据集 | 图片数 | 格式 | 下载方式 | 覆盖类别 |
|--------|--------|------|---------|---------|
| 风电叶片表面缺陷数据集 | 3,800+ | YOLO txt | 百度网盘 | 待确认 |
| Blade30 无人机巡检数据集 | 1,302 | JSON+PNG | Google Drive | Defects, Contaminations |
| 风电叶片分类数据集 | 1,000+ | VOC XML | GitHub | crack, thunderstrike等6类 |

## 参考文献

1. SOD-YOLO (2022): 改进YOLOv5+CBAM, mAP 95.1%
2. WTBD-YOLOv8 (2024): YOLOv8+GhostCBS+MHSA, AP 98.3%
3. GCB-YOLO (2025): YOLOv5s+GhostNet+CA+BiFPN, mAP 94.72%
4. LE-YOLO (2024): YOLOv7+GSConv+SimAM, mAP 78.7%
5. Memari et al. (2024): 无人机+深度学习检测综述, 114引用

## License

MIT License
