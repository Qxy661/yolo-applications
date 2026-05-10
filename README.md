# YOLO + VisDrone 低空小目标检测

基于 YOLOv8 的低空无人机目标检测，使用 VisDrone2019-DET 数据集。

## 项目结构

```
├── download_data.py          # 数据集下载与格式转换
├── setup_env.bat             # 一键环境配置
├── requirements.txt          # Python 依赖
├── data/visdrone/            # 数据集
│   ├── visdrone.yaml         # YOLO 数据集配置
│   ├── images/               # 图片 (train/val/test)
│   └── labels/               # YOLO 标注
├── src/
│   ├── train.py              # 训练脚本
│   ├── detect.py             # 推理脚本
│   ├── evaluate.py           # 评估脚本
│   ├── visualize.py          # 可视化脚本
│   ├── sahi_eval.py          # SAHI 切片推理评估
│   ├── sahi_detect.py        # SAHI 检测可视化
│   └── optimize.py           # 零训练优化实验
├── results/                  # 实验结果与指标 JSON
├── docs/                     # 项目文档 (8 篇)
└── ppt/                      # 结题汇报
```

## 快速开始

```bash
# 1. 环境配置
setup_env.bat

# 2. 下载数据集
python download_data.py

# 3. 训练基线模型
python src/train.py --model yolov8n --epochs 100

# 4. 评估
python src/evaluate.py --weights runs/baseline/yolov8n_visdrone/weights/best.pt

# 5. 可视化
python src/visualize.py detect --weights runs/baseline/yolov8n_visdrone/weights/best.pt --source data/visdrone/images/val
```

## 文档导航

| 文档 | 说明 | 适合谁 |
|------|------|--------|
| [使用教程](docs/usage_guide.md) | 从零复现项目的完整步骤 | 想跑通项目的人 |
| [YOLO 保姆级详解](docs/yolo_deep_dive.md) | YOLOv8 架构、损失函数、训练/推理全链路深度解析 | 想深入理解 YOLO 的人 |
| [学习认知总结](docs/learning_summary.md) | 实验反思、踩坑记录、技术决策背后的原因 | 想了解项目思路的人 |
| [YOLO 全面认知](docs/yolo_guide.md) | YOLO 系列发展历程与核心概念概览 | 快速了解 YOLO 全貌 |
| [论文调研](docs/paper_review.md) | CBAM、SAHI、Dynamic Head 三篇论文 | 做文献调研的人 |
| [实验记录](docs/experiment_log.md) | 四轮实验的详细过程与数据 | 需要实验细节的人 |
| [技术报告](docs/technical_report.md) | 项目技术总结，含完整实验结果 | 写报告/论文的人 |
| [结题汇报 PPT](ppt/presentation.md) | 13 页汇报内容 | 答辩/汇报用 |

## 实验结果总览

| 实验 | 方法 | mAP@0.5 | vs 基线 |
|------|------|---------|---------|
| 基线 | YOLOv8n + 640px | 0.2979 | - |
| 改进 | YOLOv8s + 800px + 强增强 | 0.4258 | +43.0% |
| SAHI | 切片推理 (640×640, 重叠20%) | 0.4523 | +51.8% |
| **零训练优化** | **SAHI + conf=0.05** | **0.4903** | **+64.6%** |

## 类别

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
