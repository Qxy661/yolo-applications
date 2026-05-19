# 风电场叶片缺陷检测 — 文献综述

## 1. 研究背景

风力发电机叶片长期暴露在恶劣环境中，面临风沙侵蚀、雷击、紫外线老化等威胁。叶片缺陷若不及时检测，可能导致停机甚至安全事故。传统人工巡检效率低、成本高、主观性强，基于深度学习的自动化检测成为主流方向。

## 2. YOLO系列在叶片检测中的应用

### 2.1 主要研究成果

| 论文 | 年份 | 基线模型 | 改进方法 | 关键结果 |
|------|------|---------|---------|---------|
| SOD-YOLO (Zhang & Wen) | 2022 | YOLOv5 | CBAM注意力 + 微尺度检测 | mAP 95.1%, +7.82% |
| WTBD-YOLOv8 (Tong et al.) | 2024 | YOLOv8 | GhostCBS + MHSA + Mini-BiFPN | AP 98.3%, 参数↓38.2% |
| LE-YOLO (Fu et al.) | 2024 | YOLOv7 | GSConv + SimAM + EIoU | mAP 78.7%, 105.1 FPS |
| GCB-YOLO (Zhang et al.) | 2025 | YOLOv5s | GhostNet + CA + BiFPN | mAP@0.5 94.72%, 7.5MB |
| Davis et al. | 2024 | YOLOv9-C | ResNet18-FPN | mAP50 0.849 |

### 2.2 技术趋势

1. **轻量化**: GhostNet/GSConv替代标准卷积，参数量↓30-50%
2. **多尺度融合**: BiFPN/PAFPN替代标准FPN
3. **注意力机制**: CA/SimAM优于CBAM（小缺陷场景）
4. **数据增强**: Mosaic+MixUp+CopyPaste组合效果最佳
5. **切片推理**: SAHI对小缺陷检测提升10-15%

## 3. 注意力机制对比

| 注意力 | 类型 | 参数开销 | 位置感知 | 小缺陷效果 | 推荐场景 |
|--------|------|---------|---------|-----------|---------|
| SE | 通道 | 极低 | 无 | 一般 | 通用检测 |
| CBAM | 通道+空间 | 低 | 弱 | 可能有害 | 大中目标 |
| CA | 通道+位置 | 低 | 有 | 推荐 | 小缺陷检测 |
| ECA | 通道(轻量) | 极低 | 无 | 轻量推荐 | 边缘部署 |
| SimAM | 无参数 | 无 | 有 | 推荐 | 轻量场景 |
| C2PSA | 位置敏感注意力 | 中 | 有 | 推荐 | YOLOv11+ |

**关键发现**: CBAM的空间注意力可能抑制小缺陷特征（同VisDrone小目标实验结论），建议优先使用CA或ECA。

## 4. 数据集综述

### 4.1 主要公开数据集

| 数据集 | 图片数 | 类别数 | 标注格式 | 来源 |
|--------|--------|--------|---------|------|
| WTB Dataset | ~2,000 | 多类 | YOLO | SOD-YOLO论文 |
| WTBDD | ~3,000 | 多类 | YOLO | WTBD-YOLOv8论文 |
| 9类UAV风电数据集 | 4,467 | 9 | YOLO | GitHub |
| Kaggle风电数据集 | ~308 | 5 | VOC | Kaggle |
| 7类无人机数据集 | ~500 | 7 | YOLO | GitHub |

### 4.2 五类缺陷覆盖情况

| 目标类别 | 最佳数据来源 | 覆盖质量 |
|---------|-------------|---------|
| 叶片裂纹 (Crack) | 9类UAV数据集 | 优秀 |
| 叶片破损 (Breakage) | Kaggle数据集 | 良好 |
| 雷击损伤 (Lightning) | 7类数据集(Burn类) | 一般 |
| 涂层脱落 (Peeling) | 9类UAV数据集 | 优秀 |
| 边缘侵蚀 (Erosion) | 9类UAV数据集 | 良好 |

## 5. 本项目技术选型

### 5.1 模型选择: YOLOv11
- **理由**: ultralytics最新版本，架构成熟，社区支持好
- **基线**: YOLOv11n (nano) 轻量化基线
- **改进方向**: YOLOv11s (small) + CA注意力 + BiFPN

### 5.2 注意力机制: CA (Coordinate Attention)
- **理由**: 保留位置信息，对小缺陷友好，参数量低
- **备选**: ECA (极轻量) / SimAM (无参数)

### 5.3 数据增强策略
- Mosaic (概率1.0) + MixUp (概率0.15) + CopyPaste (概率0.3)
- 颜色抖动 + 几何变换

### 5.4 推理优化
- SAHI切片推理 (已验证有效)
- ONNX/TensorRT导出

## 6. 参考文献

1. Zhang & Wen, "SOD-YOLO: Small Object Detection YOLO Based on Improved YOLOv5", 2022
2. Tong et al., "WTBD-YOLOv8: Wind Turbine Blade Defect Detection", 2024
3. Fu et al., "LE-YOLO: Enhanced YOLO for Leading Edge Defect Detection", 2024
4. Zhang et al., "GCB-YOLO: GhostNet+CA+BiFPN for Blade Detection", 2025
5. Memari et al., "Wind Turbine Blade Defect Detection with YOLO Models", 2024
6. Masita et al., "Deep Learning for WTB Defect Detection: A Review", 2025
7. Wang et al., "Lightning Strike Damage Detection for Wind Turbines", 2022
8. Rabbi et al., "VR-generated Synthetic Data for Blade Defect Detection", 2023
