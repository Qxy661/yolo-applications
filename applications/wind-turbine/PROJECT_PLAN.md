# 风电场叶片缺陷检测 — 详细项目计划

## 1. 项目背景与目标

### 1.1 研究背景
风力发电机叶片长期暴露在恶劣环境中，面临风沙侵蚀、雷击、紫外线老化等威胁。叶片缺陷若不及时检测，可能导致停机甚至安全事故。传统人工巡检效率低、成本高、主观性强，基于深度学习的自动化检测成为主流方向。

### 1.2 研究目标
- 构建支持 **5 类核心缺陷** 的检测系统
- 使用 **YOLOv11** 作为基线检测器
- 目标 mAP@0.5 > 0.80（vs 基线提升 +10%）
- 支持无人机巡检图像输入
- 模型轻量化，支持边缘部署

### 1.3 五类核心缺陷定义

| 类别ID | 名称 | 英文 | 视觉特征 | 检测难度 |
|--------|------|------|---------|---------|
| 0 | 叶片裂纹 | Crack | 细长线条状，宽度<1mm | ★★★★★ |
| 1 | 叶片破损 | Breakage | 不规则缺损区域 | ★★★ |
| 2 | 雷击损伤 | Lightning | 烧蚀痕迹，碳化变色 | ★★★★ |
| 3 | 涂层脱落 | Peeling | 表面涂层剥离，露出基材 | ★★★ |
| 4 | 边缘侵蚀 | Erosion | 前缘磨损，厚度减薄 | ★★★★ |

## 2. 文献综述

### 2.1 YOLO系列在叶片检测中的应用

| 论文 | 年份 | 方法 | 关键结果 |
|------|------|------|---------|
| SOD-YOLO | 2022 | 改进YOLOv5 + CBAM + 微尺度检测 | mAP 95.1%, +7.82% |
| WTBD-YOLOv8 | 2024 | YOLOv8 + GhostCBS + MHSA + Mini-BiFPN | AP 98.3%, 参数↓38.2% |
| LE-YOLO | 2024 | 增强YOLOv7 + GSConv + SimAM + EIoU | mAP 78.7%, 105.1 FPS |
| GCB-YOLO | 2025 | YOLOv5s + GhostNet + CA + BiFPN | mAP@0.5 94.72%, 7.5MB |
| Davis et al. | 2024 | YOLOv9-C + ResNet18-FPN | mAP50 0.849 |

### 2.2 注意力机制对比

| 注意力 | 类型 | 参数开销 | 位置感知 | 小缺陷效果 | 推荐 |
|--------|------|---------|---------|-----------|------|
| SE | 通道 | 极低 | 无 | 一般 | 通用 |
| CBAM | 通道+空间 | 低 | 弱 | 可能有害 | 大中目标 |
| CA | 通道+位置 | 低 | 有 | 推荐 | 小缺陷 |
| ECA | 通道(轻量) | 极低 | 无 | 轻量推荐 | 边缘部署 |
| SimAM | 无参数 | 无 | 有 | 推荐 | 轻量场景 |

**关键发现**：CBAM的空间注意力可能抑制小缺陷特征（同VisDrone小目标实验结论），建议优先使用CA或ECA。

### 2.3 关键技术趋势
1. **轻量化**：GhostNet/ GSConv 替代标准卷积，参数量↓30-50%
2. **多尺度融合**：BiFPN/PAFPN 替代标准FPN，小缺陷检测提升显著
3. **注意力机制**：CA/SimAM 优于 CBAM（小缺陷场景）
4. **数据增强**：Mosaic + MixUp + CopyPaste 组合效果最佳
5. **切片推理**：SAHI 对小缺陷检测提升 10-15%（同低空小目标结论）

## 3. 数据集方案

### 3.1 数据集来源

| 数据集 | 图片数 | 类别 | 格式 | 来源URL |
|--------|--------|------|------|---------|
| 9类UAV风电数据集 | 4,467 | 裂纹/侵蚀/脱漆等9类 | YOLO txt | GitHub QQ767172261 |
| Kaggle风电数据集 | ~308 | 裂纹/侵蚀/破损/孔洞/前缘 | VOC XML | Kaggle |
| 7类无人机数据集 | ~500 | 裂纹/燃烧/变形/污垢/油渍/剥落/锈 | YOLO txt | GitHub QQ767172261 |

### 3.2 类别映射与合并策略

```
目标5类 ← 源数据集类别映射:
├── 0: Crack (裂纹) ← Crack
├── 1: Breakage (破损) ← Damaged, Broken, Hole, Chip
├── 2: Lightning (雷击) ← Burn (燃烧)
├── 3: Peeling (脱漆) ← Peeling, Flaking, Paint
└── 4: Erosion (侵蚀) ← Erosion, LE-Erosion
```

### 3.3 数据集划分
- 训练集: 验证集: 测试集 = 8:1:1
- 数据增强: Mosaic(概率1.0) + MixUp(概率0.15) + CopyPaste(概率0.3)
- 输入尺寸: 640×640 (标准), 1280×1280 (高精度)

## 4. 技术方案

### 4.1 基线模型选择
- **模型**: YOLOv11n (nano) / YOLOv11s (small)
- **预训练权重**: COCO 预训练
- **输入尺寸**: 640×640
- **训练轮次**: 100 epochs
- **批量大小**: 16 (YOLOv11n), 8 (YOLOv11s)
- **优化器**: AdamW, lr=0.001

### 4.2 改进方案（迭代优化）

#### 迭代一：数据增强优化
- 增强策略: Mosaic + MixUp + CopyPaste + 颜色抖动
- 预期提升: +5-10%

#### 迭代二：注意力机制集成
- 在C2f模块后添加CA/ECA注意力
- 预期提升: +2-3%

#### 迭代三：Neck增强
- 替换标准FPN为BiFPN/PAFPN
- 预期提升: +3-5%

#### 迭代四：SAHI切片推理
- 切片尺寸: 640, overlap: 0.2, conf: 0.2
- 预期提升: +10-15%

### 4.3 评估指标
- 主要指标: mAP@0.5, mAP@0.5:0.95
- 辅助指标: Precision, Recall, F1-Score
- 效率指标: FPS, 参数量(M), FLOPs(G), 模型大小(MB)

## 5. 实验计划

### 第一阶段：基线建立 (第1周)
1. 下载并合并数据集
2. 数据格式统一为YOLO格式
3. YOLOv11n 基线训练
4. 基线评估与分析

### 第二阶段：数据增强 (第2周)
1. 测试不同增强策略组合
2. 对比Mosaic/MixUp/CopyPaste效果
3. 优化超参数

### 第三阶段：架构改进 (第3-4周)
1. 集成CA/ECA注意力机制
2. 尝试BiFPN/PAFPN Neck
3. 消融实验

### 第四阶段：推理优化 (第5周)
1. SAHI切片推理集成
2. ONNX/TensorRT导出
3. 推理速度优化

### 第五阶段：论文撰写 (第6-8周)
1. 实验结果整理
2. 消融实验表格
3. 论文/报告撰写

## 6. 依赖环境

```
Python 3.10+
PyTorch 2.6.0+cu124
ultralytics 8.4.51
opencv-python
matplotlib
numpy
pillow
pyyaml
tqdm
sahi (for slice inference)
```

## 7. 参考文献

1. Zhang & Wen, "SOD-YOLO: Small Object Detection YOLO Based on Improved YOLOv5", 2022
2. Tong et al., "WTBD-YOLOv8: Wind Turbine Blade Defect Detection", 2024
3. Fu et al., "LE-YOLO: Enhanced YOLO for Leading Edge Defect Detection", 2024
4. Zhang et al., "GCB-YOLO: GhostNet+CA+BiFPN for Blade Detection", 2025
5. Memari et al., "Wind Turbine Blade Defect Detection with YOLO Models", 2024
6. Masita et al., "Deep Learning for WTB Defect Detection: A Review", 2025
7. Wang et al., "Lightning Strike Damage Detection for Wind Turbines", 2022
8. Rabbi et al., "VR-generated Synthetic Data for Blade Defect Detection", 2023
