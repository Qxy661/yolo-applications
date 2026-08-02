# YOLO 小目标检测 🎯

> 基于 YOLO 系列的低空/工业场景小目标检测项目，覆盖多个应用领域。
> 以**应用流程闭环**（数据→训练→评估→部署）为组织主线，可复用于任何检测任务。

**🔬 最佳效果：YOLOv8s + 800px + 强增强 → SAHI切片推理 → mAP@0.5 = 0.4903（+64.6% vs Baseline）**

---

## 闭环全景

```
┌─ 数据准备 ──────────────────────────────────────┐
│ VisDrone / SteelBall / WindTurbine              │
│ → 获取 → 清洗 → 转换 → 划分 → data.yaml          │
├─ 训练微调 ──────────────────────────────────────┤
│ 基线(COCO预训练) → 微调 → best.pt → 调参优化     │
├─ 评估优化 ──────────────────────────────────────┤
│ mAP / AP_small → 阈值搜索 / SAHI 切片            │
├─ 部署上线 ──────────────────────────────────────┤
│ ONNX / TensorRT → 实时推理 → 业务集成            │
└────────────────────────────────────────────────┘
```

## 应用场景

| 场景 | 基线模型 | 数据集 | 核心技术 | 状态 |
|------|---------|--------|---------|------|
| [低空无人机检测](#低空无人机检测-visdrone) | YOLOv8n/s + **YOLO26** | VisDrone2019-DET (10类) | P2检测头 + CBAM + SAHI | ✅ 实验完成 |
| [钢珠检测（电赛）](applications/steel-ball/) | YOLO26n | 钢珠 1类 (1943张) | json→yolo转换 + 清洗 | 🔄 进行中 |
| [风电叶片缺陷检测](applications/wind-turbine/) | YOLOv8n/v11n | Blade30 + WT | COCO预训练 + 微调 + SAM | ✅ 已完成 |

## 项目结构

```
yolo-visdrone/
├── src/              # 核心代码（训练/评估/SAHI/优化）
├── scripts/          # 应用脚本（钢珠训练/部署）
├── data/             # 数据集配置
├── docs/             # 文档（tutorial 知识体系 + 工程文档）
├── applications/     # 具体应用（steel-ball / wind-turbine）
├── results/          # 实验指标
└── runs/             # 训练输出（gitignored）
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练（以钢珠为例）
python scripts/train_steelball.py --model yolo26n.pt

# 部署
python scripts/deploy_yolo.py export --model best.pt
python scripts/deploy_yolo.py infer --model best.onnx --source 0
```

## 低空无人机检测 (VisDrone)

### 快速复现最佳效果
```bash
# 1. 环境配置
pip install -r requirements.txt
# 2. 使用已训练好的最佳模型推理
yolo predict model=weights/best.pt source=...
# 3. SAHI 切片推理（零成本提升小目标）
python src/sahi_detect.py ...
```

### 实验结果

（最新 YOLO26 结果训练完成后填入）

| 方法 | mAP@0.5 | mAP@0.5:0.95 |
|------|---------|-------------|
| Baseline (YOLOv8n) | 0.298 | 0.166 |
| Improved (YOLOv8s) | 0.426 | 0.245 |
| SAHI (Improved) | 0.490 | 0.268 |
| **YOLO26 (最新)** | **训练中...** | — |

### VisDrone 类别
pedestrian / people / bicycle / car / van / truck / tricycle / awning-tricycle / bus / motor

## 风电叶片缺陷检测

> 详细文档见 [applications/wind-turbine/](applications/wind-turbine/)

基于 YOLOv11 的风电叶片表面缺陷自动检测系统，支持 5 类核心缺陷识别。

| 类别ID | 中文名称 | 英文名称 |
|--------|---------|---------|
| 0 | 叶片裂纹 | Crack |
| 1 | 叶片破损 | Breakage |
| 2 | 雷击损伤 | Lightning |
| 3 | 涂层脱落 | Peeling |
| 4 | 边缘侵蚀 | Erosion |

```bash
cd applications/wind-turbine
python src/train.py --model yolo11n.pt --epochs 100 --batch 16
```

**核心技术路线**：YOLOv11 + GhostNet轻量化 + CA坐标注意力 + BiFPN + SAHI + 分辨率渐进训练

## 文档导航

- **知识体系**（[docs/tutorial/](docs/tutorial/00-README.md)）：从原理到应用的 7 篇教程
- **工程文档**（[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)、[docs/TECHNICAL_SUMMARY.md](docs/TECHNICAL_SUMMARY.md)）
- **贡献指南**（[CONTRIBUTING.md](CONTRIBUTING.md)）

## 参考文献

1. VisDrone2019-DET Dataset: Zhu et al., "Vision Meets Drones: A Challenge", arXiv 2018
2. SAHI: Akyon et al., "Slicing Aided Hyper Inference", WACV 2023
3. SOD-YOLO (2022): Zhang & Wen, mAP 95.1%
4. WTBD-YOLOv8 (2024): Tong et al., AP 98.3%

## License

MIT © [Qxy661](https://github.com/Qxy661)
