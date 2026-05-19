# YOLO 小目标检测

基于 YOLO 系列的低空/工业场景小目标检测项目，覆盖多个应用领域。

## 应用场景

| 场景 | 基线模型 | 数据集 | 核心技术 | 状态 |
|------|---------|--------|---------|------|
| [低空无人机检测](#低空无人机检测-visdrone) | YOLOv8n/s | VisDrone2019-DET (10类) | P2检测头 + CBAM + SAHI | 已完成4轮实验 |
| [风电叶片缺陷检测](applications/wind-turbine/) | YOLOv11n/s | 多源合并 (5类) | CA注意力 + GhostNet + BiFPN | 调研完成，待训练 |

---

## 项目结构

```
YOLO-VisDrone/
├── src/                              # 通用代码 (VisDrone)
│   ├── train.py / train_p2.py        # 训练脚本
│   ├── detect.py / evaluate.py       # 推理与评估
│   ├── sahi_detect.py / sahi_eval.py # SAHI 切片推理
│   ├── cbam.py                       # CBAM 注意力模块
│   └── register_custom_modules.py    # 自定义模块注册
├── configs/                          # 模型配置
│   └── yolov8s-p2.yaml
├── data/visdrone/                    # VisDrone 数据集
├── docs/                             # 项目文档
│   ├── experiments/                  # 实验报告
│   └── ...技术文档...
├── ppt/                              # 汇报材料
├── applications/                     # 应用场景扩展
│   └── wind-turbine/                 # 风电叶片缺陷检测
│       ├── src/                      # 风电专用脚本
│       ├── configs/                  # YOLOv11 配置
│       ├── scripts/                  # 数据集工具
│       ├── data/                     # 风电数据配置
│       └── docs/                     # 风电技术文档
├── requirements.txt                  # 统一依赖
└── README.md
```

---

## 低空无人机检测 (VisDrone)

基于 VisDrone2019-DET 数据集的低空小目标检测，解决无人机航拍图像中行人、车辆等小目标检测难题。

### 快速开始

```bash
# 环境配置
setup_env.bat

# 下载数据集
python download_data.py

# 训练基线
python src/train.py --model yolov8n --epochs 100

# 评估
python src/evaluate.py --weights runs/baseline/yolov8n_visdrone/weights/best.pt

# SAHI 切片推理
python src/sahi_eval.py --weights runs/baseline/yolov8n_visdrone/weights/best.pt
```

### 实验结果

| 实验 | 方法 | mAP@0.5 | vs 基线 |
|------|------|---------|---------|
| 基线 | YOLOv8n + 640px | 0.2979 | - |
| 改进 | YOLOv8s + 800px + 强增强 | 0.4258 | +43.0% |
| SAHI | 切片推理 (640×640, 重叠20%) | 0.4523 | +51.8% |
| **零训练优化** | **SAHI + conf=0.05** | **0.4903** | **+64.6%** |
| P2+CBAM | P2 检测头 + CBAM 注意力 | 训练中 | - |

### VisDrone 类别

| ID | 类别 | 中文 |
|----|------|------|
| 0 | pedestrian | 行人 |
| 1 | people | 人群 |
| 2 | bicycle | 自行车 |
| 3 | car | 汽车 |
| 4 | van | 面包车 |
| 5 | truck | 卡车 |
| 6 | tricycle | 三轮车 |
| 7 | awning-tricycle | 篷三轮车 |
| 8 | bus | 公交车 |
| 9 | motor | 摩托车 |

### 文档导航

| 文档 | 说明 |
|------|------|
| [使用教程](docs/usage_guide.md) | 从零复现的完整步骤 |
| [YOLO 保姆级详解](docs/yolo_deep_dive.md) | YOLOv8 架构、损失函数全链路解析 |
| [YOLO 全面认知](docs/yolo_guide.md) | YOLO 系列发展历程概览 |
| [论文调研](docs/paper_review.md) | CBAM、SAHI、Dynamic Head 三篇论文 |
| [实验报告](docs/experiments/) | 4轮实验详细记录 |
| [技术报告](docs/technical_report.md) | 项目技术总结 |
| [结题汇报](ppt/presentation.md) | 13页汇报内容 |

---

## 风电叶片缺陷检测

> 详细文档见 [applications/wind-turbine/](applications/wind-turbine/)

基于 YOLOv11 的风电叶片表面缺陷自动检测系统，支持 5 类核心缺陷识别。

### 风电类别

| 类别ID | 中文名称 | 英文名称 |
|--------|---------|---------|
| 0 | 叶片裂纹 | Crack |
| 1 | 叶片破损 | Breakage |
| 2 | 雷击损伤 | Lightning |
| 3 | 涂层脱落 | Peeling |
| 4 | 边缘侵蚀 | Erosion |

### 快速开始

```bash
cd applications/wind-turbine

# 训练基线
python src/train.py --model yolo11n.pt --epochs 100 --batch 16

# 改进训练 (带CA注意力)
python src/train_improved.py --model yolo11n.pt --use-ca --epochs 100
```

### 核心技术路线

YOLOv11 + GhostNet轻量化 + CA坐标注意力 + BiFPN特征融合 + SAHI切片推理 + 分辨率渐进训练

### 调研成果

- 16+篇核心论文深度分析（SOD-YOLO/WTBD-YOLOv8/LE-YOLO/GCB-YOLO等）
- 10+个开源项目评估（DeepCrack 355★/FPHB 408★等）
- 风电+桥梁双领域综合调研报告

---

## 参考文献

### 低空检测
1. VisDrone2019-DET Dataset: Zhu et al., "Vision Meets Drones: A Challenge", arXiv 2018
2. SAHI: Akyon et al., "Slicing Aided Hyper Inference", WACV 2023

### 风电检测
1. SOD-YOLO (2022): Zhang & Wen, mAP 95.1%, DOI: 10.1002/adts.202100631
2. WTBD-YOLOv8 (2024): Tong et al., AP 98.3%, DOI: 10.3390/su16114467
3. GCB-YOLO (2025): Zhang et al., mAP 94.72%, DOI: 10.1002/we.70029
4. LE-YOLO (2024): Fu et al., mAP 78.7%, DOI: 10.1109/access.2024.3463391

## License

MIT License
