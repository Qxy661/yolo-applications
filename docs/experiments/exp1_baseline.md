# 实验一：基线实验

> YOLOv8n + VisDrone2019-DET Baseline
> 实验日期: 2026-05-12
> 实验者: 小羊

---

## 1. 实验目标

建立 VisDrone 数据集上的检测基准线（Baseline），为后续改进实验提供对比参照。

**核心问题**: YOLOv8n（最轻量模型）在无人机航拍小目标数据集上的表现如何？

---

## 2. 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| Python | 3.9.12 |
| PyTorch | 2.6.0+cu124 |
| Ultralytics | 8.4.48 |
| 操作系统 | Windows 11 |
| CUDA | 12.4 |

---

## 3. 数据集

### 3.1 数据集概况

| 项目 | 说明 |
|------|------|
| 数据集 | VisDrone2019-DET |
| 来源 | 天津大学无人机视觉实验室 |
| 训练集 | 6,471 张图片 |
| 验证集 | 548 张图片 |
| 图片分辨率 | ~2000×1500 像素 |
| 类别数 | 10 类 |

### 3.2 类别定义

| ID | 英文名 | 中文名 | 典型像素尺寸 |
|----|--------|--------|-------------|
| 0 | pedestrian | 行人 | 10~30px |
| 1 | people | 人群 | 20~50px |
| 2 | bicycle | 自行车 | 15~35px |
| 3 | car | 汽车 | 30~80px |
| 4 | van | 面包车 | 40~100px |
| 5 | truck | 卡车 | 50~120px |
| 6 | tricycle | 三轮车 | 20~50px |
| 7 | awning-tricycle | 篷三轮车 | 25~60px |
| 8 | bus | 公交车 | 60~150px |
| 9 | motor | 摩托车 | 15~35px |

### 3.3 数据预处理

**格式转换**: VisDrone 原始标注 → YOLO 格式

```
# VisDrone 原始格式
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

# YOLO 格式（归一化）
<class_id> <x_center> <y_center> <width> <height>
```

**转换规则**:
- 坐标归一化: `x_center = (x + w/2) / img_width`
- 类别映射: VisDrone 1~10 → YOLO 0~9
- 过滤: 忽略类别 0（ignored regions）和 score < 0 的标注

---

## 4. 模型架构

### 4.1 YOLOv8n 结构

YOLOv8n (Nano) 是 YOLOv8 系列中最轻量的变体，包含三个主要组件：

| 组件 | 功能 | 关键模块 |
|------|------|---------|
| Backbone | 特征提取 | CSPDarknet + C2f 模块 |
| Neck | 多尺度特征融合 | PANet + FPN |
| Head | 检测输出 | Decoupled Head, Anchor-Free |

### 4.2 模型参数

| 参数 | 值 |
|------|-----|
| 参数量 | 3.2M |
| FLOPs | 8.7G |
| 层数 | 225 |
| 检测层 | P3(80×80) + P4(40×40) + P5(20×20) |
| 检测点数 | 8,400 |
| COCO mAP@0.5:0.95 | 37.3% |

### 4.3 核心创新点

1. **C2f 模块**: 跨阶段部分连接，梯度流更丰富
2. **Decoupled Head**: 分类和回归独立分支，解耦优化
3. **Anchor-Free**: 直接预测目标中心和宽高，无需预定义锚框
4. **DFL Loss**: 分布式焦点损失，提升边界框回归精度

---

## 5. 训练配置

### 5.1 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | YOLOv8n | Nano 变体 |
| 预训练权重 | yolov8n.pt (COCO) | 迁移学习 |
| 输入尺寸 | 640×640 | 默认分辨率 |
| Batch Size | 16 | 受限于 8GB VRAM |
| Epochs | 50 | 基础训练轮数 |
| 学习率 lr0 | 0.01 | 初始学习率 |
| 学习率 lrf | 0.01 | 最终学习率 = lr0 × lrf |
| 优化器 | SGD (auto) | 自动选择 |
| 混合精度 | AMP (FP16) | 加速训练 |
| 权重衰减 | 0.0005 | 正则化 |
| 动量 | 0.937 | SGD 动量 |

### 5.2 数据增强

| 增强方法 | 参数 | 作用 |
|---------|------|------|
| Mosaic | 1.0 (默认) | 4 图拼接，增加上下文多样性 |
| HSV-H | 0.015 | 色调抖动 |
| HSV-S | 0.7 | 饱和度抖动 |
| HSV-V | 0.4 | 亮度抖动 |
| 翻转 | fliplr=0.5 | 水平随机翻转 |
| 缩放 | scale=0.5 | 随机缩放 |
| 旋转 | degrees=0.0 | 无旋转（默认） |
| 平移 | translate=0.1 | 随机平移 |

---

## 6. 训练过程

### 6.1 训练耗时

| 项目 | 值 |
|------|-----|
| 总耗时 | ~65 分钟 |
| 每 Epoch | ~1.3 分钟 |
| 每 Iteration | ~0.1 秒 |

### 6.2 损失曲线

训练损失从初始高位逐步收敛：
- **Box Loss**: 边界框回归损失（CIoU）
- **Cls Loss**: 分类损失（BCE）
- **DFL Loss**: 分布式焦点损失

（详见 `runs/detect/runs/baseline/yolov8n_visdrone/results.png`）

### 6.3 混淆矩阵

（详见 `runs/detect/runs/baseline/yolov8n_visdrone/confusion_matrix.png`）

---

## 7. 评估结果

### 7.1 评估配置

| 参数 | 值 |
|------|-----|
| 评估集 | val (548 张) |
| 置信度阈值 | 0.25 |
| NMS IoU 阈值 | 0.6 |
| 评估工具 | Ultralytics val + pycocotools |

### 7.2 整体指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **mAP@0.5** | **0.2979** | IoU=0.5 时的平均精度 |
| **mAP@0.5:0.95** | **0.1663** | IoU=0.5~0.95 的平均精度 |
| Precision | 0.4237 | 精确率 |
| Recall | 0.3191 | 召回率 |

### 7.3 各类别 AP@0.5

| 类别 | AP@0.5 | 排名 | 分析 |
|------|--------|------|------|
| car | 0.723 | 1 | 大目标，检测效果好 |
| bus | 0.395 | 2 | 大目标，但样本少 |
| van | 0.337 | 3 | 中等目标 |
| motor | 0.329 | 4 | 小目标，形状规则 |
| pedestrian | 0.312 | 5 | 小目标，样本多 |
| truck | 0.278 | 6 | 大目标，但样本少 |
| people | 0.243 | 7 | 小目标，边界模糊 |
| tricycle | 0.199 | 8 | 稀有类别 |
| awning-tricycle | 0.100 | 9 | 最稀有类别 |
| bicycle | 0.062 | 10 | 最难检测 |

---

## 8. 结果分析

### 8.1 关键发现

1. **整体精度偏低**: mAP@0.5 = 29.79%，远低于 COCO 上的 37.3%，说明 VisDrone 比 COCO 更具挑战性
2. **大目标检测尚可**: car (0.723) 是唯一超过 0.5 的类别
3. **小目标检测极差**: bicycle (0.062)、awning-tricycle (0.100) 几乎无法检测
4. **召回率低**: Recall = 31.91%，大量目标被漏检

### 8.2 原因分析

| 问题 | 原因 |
|------|------|
| 小目标漏检 | YOLOv8n 最小检测层 P5=20×20，无法捕获 <16px 目标 |
| 输入分辨率低 | 640×640 导致 2000×1500 原图中的小目标信息丢失 |
| 模型容量不足 | 3.2M 参数对 10 类复杂场景表达能力有限 |
| 类别不平衡 | bicycle/awning-tricycle 样本极少 |

### 8.3 改进方向

基于以上分析，后续改进应聚焦：

1. **模型升级**: 使用更大容量的 YOLOv8s (11.2M 参数)
2. **增大输入**: 从 640 提升到 800，保留更多小目标像素
3. **增强数据**: MixUp、CopyPaste 等增强策略
4. **架构改进**: 添加 P2 小目标检测头

---

## 9. 复现说明

### 9.1 复现命令

```bash
# 确保在项目根目录
cd E:\yolo-visdrone

# 运行基线训练
python src/train_baseline.py
```

### 9.2 输出文件

| 文件 | 路径 |
|------|------|
| 模型权重 | `runs/detect/runs/baseline/yolov8n_visdrone/weights/best.pt` |
| 训练曲线 | `runs/detect/runs/baseline/yolov8n_visdrone/results.png` |
| 混淆矩阵 | `runs/detect/runs/baseline/yolov8n_visdrone/confusion_matrix.png` |
| 评估指标 | `results/baseline_metrics.json` |

---

## 10. 结论

YOLOv8n 在 VisDrone 基线实验中 mAP@0.5 = 29.79%，暴露了三个核心问题：小目标检测能力不足、输入分辨率受限、模型容量偏小。这为后续改进实验提供了明确的优化方向。

---

*本实验为五轮递进实验的第一轮，后续实验将在此基础上逐步优化。*
