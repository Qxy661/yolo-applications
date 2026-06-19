# YOLO 小目标检测

基于 YOLO 系列的低空/工业场景小目标检测项目，覆盖多个应用领域。

**🔬 最佳效果：YOLOv8s + 800px + 强增强 → SAHI切片推理 → conf=0.05，mAP@0.5 = 0.4903（+64.6% vs Baseline）**

---

## 应用场景

| 场景 | 基线模型 | 数据集 | 核心技术 | 状态 |
|------|---------|--------|---------|------|
| [低空无人机检测](#低空无人机检测-visdrone) | YOLOv8n/s | VisDrone2019-DET (10类) | P2检测头 + CBAM + SAHI | ✅ 4轮实验完成 / ⚠️ P2+CBAM 训练中 |
| [风电叶片缺陷检测](applications/wind-turbine/) | YOLOv8n/v11n | Blade30 + WT (2类/5类) | COCO预训练 + 全参微调 + SAM | ✅ 基线+对比完成 / 架构改进待续 |

---

## 项目结构

```
YOLO-VisDrone/
├── src/                              # 训练/推理/评估脚本
│   ├── train.py                      # Baseline 训练 (YOLOv8n + 640px)
│   ├── train_improved.py             # Improved 训练 (YOLOv8s + 800px + 强增强)
│   ├── train_p2.py                   # [WIP] P2+CBAM 训练
│   ├── evaluate.py / evaluate_all.py # 模型评估
│   ├── sahi_eval.py                  # SAHI 切片推理评估
│   ├── optimize.py                   # 零训练优化 (阈值搜索/TTA/集成)
│   ├── cbam.py                       # CBAM 注意力模块
│   ├── register_custom_modules.py    # 自定义模块注册
│   ├── detect.py / visualize.py      # 推理与可视化
│   └── utils.py                      # 工具函数
├── configs/                          # 模型配置
│   └── yolov8s-p2.yaml               # P2+CBAM 自定义模型结构
├── data/visdrone/                    # VisDrone 数据集
│   ├── visdrone.yaml                 # 数据配置 (10类)
│   ├── images/                       # 训练/验证/测试图片
│   └── labels/                       # YOLO 格式标注
├── runs/                             # 训练输出 (gitignored, 不上传)
│   ├── detect/runs/baseline/         # Baseline 训练输出
│   ├── detect/runs/improved/         # Improved 训练输出
│   └── p2/                           # ⚠️ P2+CBAM 训练输出
├── weights/                          # 训练好的模型权重 (可直接使用)
│   ├── best.pt                       # ✅ 最佳模型 (YOLOv8s, mAP 0.4903)
│   ├── baseline.pt                   # Baseline 对照
│   └── README.md                     # 权重说明
├── docs/                             # 项目文档
│   ├── experiments/                  # 4轮实验报告
│   ├── tech_innovation_report.md     # 技术创新报告
│   ├── presentation_plan.md          # PPT 汇报方案
│   └── ...其他技术文档...
├── results/                          # 评估指标 JSON / 对比图
├── ppt/                              # 汇报材料
├── applications/wind-turbine/        # 风电叶片缺陷检测 (完整子项目)
│   ├── src/configs/data/docs/...     # 独立的应用代码与数据
├── requirements.txt                  # 依赖
└── README.md
```

---

## 低空无人机检测 (VisDrone)

基于 VisDrone2019-DET 数据集的低空小目标检测，解决无人机航拍图像中行人、车辆等小目标检测难题。

### 快速复现最佳效果

```bash
# 1. 环境配置
setup_env.bat

# 2. 下载 VisDrone 数据集
python download_data.py

# 3. 使用已训练好的最佳模型推理
python src/detect.py --weights weights/best.pt --source your_image.jpg

# 4. SAHI 切片推理评估（零成本提升小目标）
python src/sahi_eval.py

# 5. 阈值搜索 + 优化（mAP@0.5 → 0.4903）
python src/optimize.py
```

**权重文件路径（推荐）：**
```bash
weights/best.pt       # 最佳模型 (YOLOv8s, mAP 0.4903)
weights/baseline.pt   # 基线对照 (YOLOv8n, mAP 0.2979)
```

**你也可以直接用训练输出路径的权重（如果是本地跑过训练）：**
- `runs/detect/runs/improved/yolov8s_visdrone/weights/best.pt`
- `runs/detect/runs/baseline/yolov8n_visdrone/weights/best.pt`

### 从头训练

```bash
# Baseline
python src/train.py --model yolov8n --epochs 50 --batch 16

# Improved（推荐）
python src/train_improved.py

# P2+CBAM [实验性，训练较慢]
python src/train_p2.py
```

### 实验结果

| 实验 | 方法 | mAP@0.5 | vs Baseline | 训练成本 |
|:----:|------|:-------:|:-----------:|:--------:|
| ① Baseline | YOLOv8n + 640px | 0.2979 | — | ~1h |
| ② Improved | YOLOv8s + 800px + 强增强 | 0.4258 | +43.0% | ~2.5h |
| ③ SAHI | 切片推理 (640×640, 20%重叠) | 0.4523 | +51.8% | 零训练 |
| ④ **零训练优化** | **SAHI + conf=0.05** | **0.4903** | **+64.6%** | **零训练** |
| ⑤ P2+CBAM | P2检测头 + CBAM注意力 | ⏳ 训练中 | — | 已训41轮 |

> P2+CBAM 为架构探索实验，训练尚未收敛。两轮零成本优化（SAHI + 阈值搜索）已取得当前最佳效果。完整消融实验和 P2+CBAM 结论预计 7 月更新。

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
