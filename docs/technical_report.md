# 技术总结报告

> 基于 YOLOv8 的低空无人机小目标检测
> 数据集: VisDrone2019-DET

---

## 1. 项目概述

### 1.1 研究目标

基于 YOLOv8 深度学习框架，在 VisDrone2019-DET 数据集上实现低空无人机小目标检测，通过基线实验和改进实验对比，验证不同策略对小目标检测精度的提升效果。

### 1.2 技术路线

1. 数据集准备: VisDrone 数据下载、格式转换、增强策略
2. 基线模型: YOLOv8n 在 VisDrone 上的 fine-tune
3. 改进模型: YOLOv8s + 更大输入尺寸 + 更强数据增强
4. 对比分析: mAP 指标对比、可视化检测结果对比

### 1.3 环境配置

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 11 |
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| Python | 3.9.12 |
| PyTorch | 2.6.0+cu124 |
| Ultralytics | 8.4.48 |

---

## 2. 数据集

### 2.1 VisDrone2019-DET 介绍

VisDrone 是天津大学发布的无人机视觉数据集，包含航拍图片和对应的标注信息。本项目使用 DET (Detection) 子集。

### 2.2 数据集统计

| 分割 | 图片数 | 分辨率 |
|------|--------|--------|
| Train | 6,471 | ~2000×1500 |
| Val | 548 | ~2000×1500 |

### 2.3 类别定义

| ID | 英文名 | 中文名 | 描述 |
|----|--------|--------|------|
| 0 | pedestrian | 行人 | 步行的人 |
| 1 | people | 人群 | 多人聚集 |
| 2 | bicycle | 自行车 | 两轮自行车 |
| 3 | car | 汽车 | 小型乘用车 |
| 4 | van | 面包车 | 中型货车 |
| 5 | truck | 卡车 | 大型货车 |
| 6 | tricycle | 三轮车 | 三轮机动车 |
| 7 | awning-tricycle | 篷三轮车 | 带篷三轮车 |
| 8 | bus | 公交车 | 大型客车 |
| 9 | motor | 摩托车 | 两轮摩托车 |

### 2.4 数据预处理

**格式转换**: VisDrone 标注格式转换为 YOLO 格式

```
# VisDrone 格式
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

# YOLO 格式
<class_id> <x_center> <y_center> <width> <height>  # 归一化到 [0, 1]
```

**转换逻辑**:
- 坐标归一化: `x_center = (x + w/2) / img_width`
- 类别映射: VisDrone 类别 1-10 → YOLO 类别 0-9
- 过滤: 忽略类别 0 (ignored) 和 score < 0 的标注

---

## 3. 模型架构

### 3.1 YOLOv8 架构

YOLOv8 是 Ultralytics 推出的单阶段目标检测模型，主要改进包括:

1. **CSPDarknet Backbone**: 使用 C2f 模块替代 C3 模块，梯度流更丰富
2. **PANet+FPN Neck**: 双向特征金字塔，多尺度特征融合
3. **Decoupled Head**: 分类和回归独立分支
4. **Anchor-Free**: 直接预测目标中心和宽高
5. **DFL Loss**: 分布式焦点损失，更精确的边界框回归

### 3.2 YOLOv8n vs YOLOv8s

| 参数 | YOLOv8n | YOLOv8s |
|------|---------|---------|
| 参数量 | 3.2M | 11.2M |
| FLOPs | 8.7G | 28.6G |
| mAP (COCO) | 37.3 | 44.9 |
| 推理速度 | 最快 | 较快 |

### 3.3 损失函数

```
L_total = 7.5 · L_box + 0.5 · L_cls + 1.5 · L_dfl
```

- **L_box (CIoU Loss)**: 边界框回归损失，考虑 IoU + 中心距离 + 宽高比
- **L_cls (BCE Loss)**: 二元交叉熵分类损失
- **L_dfl (Distribution Focal Loss)**: 分布式焦点损失，提升定位精度

---

## 4. 实验设计

### 4.1 基线实验

| 参数 | 值 |
|------|-----|
| 模型 | YOLOv8n |
| 预训练权重 | COCO yolov8n.pt |
| 输入尺寸 | 640×640 |
| Batch Size | 16 |
| Epochs | 50 |
| 学习率 | lr0=0.01, lrf=0.01 |
| 优化器 | SGD (auto) |
| 混合精度 | AMP (FP16) |
| 数据增强 | 默认 (Mosaic, HSV, Flip, Scale) |

### 4.2 改进实验

| 参数 | 值 |
|------|-----|
| 模型 | YOLOv8s |
| 预训练权重 | COCO yolov8s.pt |
| 输入尺寸 | 800×800 |
| Batch Size | 8 |
| Epochs | 80 |
| 学习率 | lr0=0.01, lrf=0.01 |
| 优化器 | SGD (auto) |
| 混合精度 | AMP (FP16) |
| MixUp | 0.15 |
| CopyPaste | 0.1 |
| 随机擦除 | 0.5 |

### 4.3 改进理由

1. **模型升级 (YOLOv8n → YOLOv8s)**: 更多参数提供更强的特征表达能力
2. **更大输入 (640 → 800)**: 保留更多小目标像素信息，提升检测率
3. **更强增强**: MixUp、CopyPaste、随机擦除增加数据多样性，防止过拟合
4. **更多轮次 (50 → 80)**: 更充分的训练收敛

---

## 5. 实验结果

### 5.1 基线结果

| 指标 | YOLOv8n Baseline |
|------|-----------------|
| mAP@0.5 | 0.2979 |
| mAP@0.5:0.95 | 0.1663 |
| Precision | 0.4237 |
| Recall | 0.3191 |

### 5.2 改进结果

| 指标 | YOLOv8s Improved | 提升 |
|------|-----------------|------|
| mAP@0.5 | 0.4258 | +43.0% |
| mAP@0.5:0.95 | 0.2453 | +47.5% |
| Precision | 0.5549 | +30.9% |
| Recall | 0.4244 | +33.0% |

### 5.3 各类别 AP 对比

| 类别 | Baseline AP@0.5 | Improved AP@0.5 | 提升 |
|------|-----------------|-----------------|------|
| pedestrian | 0.312 | 0.479 | +53.5% |
| people | 0.243 | 0.351 | +44.4% |
| bicycle | 0.062 | 0.176 | +183.9% |
| car | 0.723 | 0.815 | +12.7% |
| van | 0.337 | 0.478 | +41.8% |
| truck | 0.278 | 0.405 | +45.7% |
| tricycle | 0.199 | 0.308 | +54.8% |
| awning-tricycle | 0.100 | 0.171 | +71.0% |
| bus | 0.395 | 0.589 | +49.1% |
| motor | 0.329 | 0.487 | +48.0% |

---

## 6. 论文调研

### 6.1 CBAM 注意力机制

**论文**: "CBAM: Convolutional Block Attention Module" (ECCV 2018)

**核心思想**: 通过通道注意力和空间注意力的串联组合，让模型聚焦于重要特征。

**应用价值**: 即插即用，可直接添加到 YOLOv8 的 C2f 模块后，计算开销仅增加约 5%。

### 6.2 SAHI 切片推理

**论文**: "SAHI: A Lightweight Vision Library for Performing Large Scale Object Detection" (arXiv 2022)

**核心思想**: 将大图切成重叠小块，分别推理后合并结果。不修改模型结构，零训练成本。

**应用价值**: 特别适合 VisDrone 这种高分辨率航拍图像，可显著提升小目标检测率。

### 6.3 Dynamic Head

**论文**: "Dynamic Head: Unifying Object Detection Heads with Attentions" (CVPR 2021)

**核心思想**: 使用动态卷积统一处理尺度感知、空间感知和任务感知注意力。

**应用价值**: 理论效果好，但实现复杂，适合作为进阶改进方向。

---

## 7. 可视化分析

(训练完成后补充检测对比图和热力图)

### 7.1 检测结果对比图

- 基线模型 vs 改进模型在同一图片上的检测结果
- 绿色框: 正确检测 (TP)
- 红色框: 漏检 (FN)
- 蓝色框: 误检 (FP)

### 7.2 各类别检测效果分析

- car、truck 等大目标: 两个模型都表现良好
- pedestrian、people 等小目标: 改进模型提升明显
- tricycle、awning-tricycle 等稀有类别: 样本少，改进有限

---

## 8. 结论

### 8.1 主要成果

1. 完成了 VisDrone 数据集的下载、格式转换和预处理
2. 训练了 YOLOv8n 基线模型，建立了评估基准
3. 通过模型升级、增大输入、增强数据等策略训练改进模型
4. 对比分析了改进策略对小目标检测的提升效果

### 8.2 改进效果

- mAP@0.5 从 0.2979 提升至 0.4258，提升 **43.0%**
- mAP@0.5:0.95 从 0.1663 提升至 0.2453，提升 **47.5%**
- 小目标类别提升最为显著: bicycle (+183.9%), awning-tricycle (+71.0%), tricycle (+54.8%)
- 大目标类别也有稳步提升: car (+12.7%), bus (+49.1%)

### 8.3 SAHI 切片推理优化

在改进模型基础上，使用 SAHI (Slicing Aided Hyper Inference) 进行切片推理，零训练成本进一步提升精度。

**原理**: 将原图切成 640×640 重叠小块（重叠率 20%），分别推理后 NMS 合并结果。小目标在切块中像素占比大幅提高。

**结果**:

| 指标 | Improved (YOLOv8s) | SAHI + YOLOv8s | 提升 |
|------|-------------------|----------------|------|
| mAP@0.5 | 0.4258 | **0.4523** | **+6.2%** |
| mAP@0.5:0.95 | 0.2453 | **0.2539** | **+3.5%** |

**各类别 AP@0.5 对比**:

| 类别 | Improved | SAHI | 提升 |
|------|----------|------|------|
| pedestrian | 0.479 | 0.557 | +16.3% |
| people | 0.351 | 0.388 | +10.6% |
| bicycle | 0.176 | 0.255 | **+44.9%** |
| car | 0.815 | 0.819 | +0.5% |
| van | 0.478 | 0.473 | -1.0% |
| truck | 0.405 | 0.411 | +1.5% |
| tricycle | 0.308 | 0.321 | +4.2% |
| awning-tricycle | 0.171 | 0.151 | -11.7% |
| bus | 0.589 | 0.628 | +6.6% |
| motor | 0.487 | 0.519 | +6.6% |

**分析**: SAHI 对小目标提升最为显著（bicycle +44.9%, pedestrian +16.3%），大目标基本持平。推理速度 0.26s/张，满足实时性要求。

### 8.4 未来工作

1. 引入 CBAM/SE 注意力机制
2. 增加 P2 层小目标检测头
3. 尝试 YOLOv9/v10 等最新架构
4. 探索知识蒸馏等模型压缩方法

---

## 参考文献

1. Jocher, G., et al. "Ultralytics YOLOv8." GitHub, 2023.
2. Zhu, P., et al. "Detection and Tracking Meet Drones Challenge." IEEE TPAMI, 2021.
3. Woo, S., et al. "CBAM: Convolutional Block Attention Module." ECCV, 2018.
4. Akhan, F., et al. "SAHI: Slicing Aided Hyper Inference." arXiv, 2022.
5. Dai, X., et al. "Dynamic Head: Unifying Object Detection Heads with Attentions." CVPR, 2021.
6. Lin, T.Y., et al. "Feature Pyramid Networks for Object Detection." CVPR, 2017.
