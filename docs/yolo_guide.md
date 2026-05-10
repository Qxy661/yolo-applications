# YOLO 全面认知文档

> 面向课程结题的 YOLO 系统性总结
> 涵盖: 发展历程、核心原理、YOLOv8 架构、训练流程、部署应用

---

## 1. YOLO 发展历程

### 1.1 版本演进

| 版本 | 年份 | 核心改进 | mAP (COCO) |
|------|------|---------|------------|
| YOLOv1 | 2016 | 单阶段检测，端到端 | 63.4 |
| YOLOv2 | 2017 | Batch Norm、Anchor Box、多尺度 | 78.6 |
| YOLOv3 | 2018 | 多尺度预测 (FPN)、Darknet-53 | 82.0 |
| YOLOv4 | 2020 | CSPDarknet、SPP、PANet | 85.1 |
| YOLOv5 | 2020 | 自适应 Anchor、Focus 层 | 87.0 |
| YOLOv8 | 2023 | Anchor-Free、Decoupled Head | 89.4 |

### 1.2 为什么 YOLO 适合小目标检测？

- **速度快**: 单阶段推理，实时检测
- **多尺度**: FPN/PANet 多尺度特征融合
- **端到端**: 不需要复杂的预处理和后处理
- **社区活跃**: 大量改进方案和预训练权重

---

## 2. YOLOv8 核心架构

### 2.1 整体结构

```
输入图像 (640×640)
       │
       ▼
┌──────────────┐
│   Backbone   │  CSPDarknet + C2f 模块
│   特征提取    │  提取多尺度特征
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     Neck     │  PANet + FPN
│  特征融合     │  自顶向下 + 自底向上
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     Head     │  Decoupled Head (Anchor-Free)
│  检测头       │  分类 + 回归 + DFL
└──────┬───────┘
       │
       ▼
    检测结果
```

### 2.2 Backbone — CSPDarknet

CSPDarknet 是 YOLOv8 的特征提取骨干网络:

```
输入 → Conv → C2f → C2f → C2f → C2f → SPPF
       │      │      │      │      │
       │      │      │      │      └─ P5 (20×20)
       │      │      │      └─ P4 (40×40)
       │      │      └─ P3 (80×80)
       │      └─ P2 (160×160)
       └─ P1 (320×320)
```

**C2f 模块**: Cross Stage Partial with 2 convolutions
- 拆分特征图 → 多个 Bottleneck → 拼接
- 比 CSPNet 更高效，保留梯度信息

### 2.3 Neck — PANet + FPN

```
P5 ──────────────────────→ P5_out
 │                           ↑
 ↓                           │
上采样 + 拼接 + C2f ────→ P4 ──→ P4_out
 │                           ↑
 ↓                           │
上采样 + 拼接 + C2f ────→ P3 ──→ P3_out
 │                           ↑
 ↓                           │
下采样 + 拼接 + C2f ────→ P4 ──┘
 │                           ↑
 ↓                           │
下采样 + 拼接 + C2f ────→ P5 ──┘
```

- **FPN (自顶向下)**: 高层语义信息向低层传递
- **PAN (自底向上)**: 低层位置信息向高层传递
- 结果: 每个尺度都有丰富的语义和位置信息

### 2.4 Head — Decoupled Head

```
特征图 → 1×1 Conv ─┬─ 分类分支 → class_pred
                     │
                     ├─ 回归分支 → box_pred
                     │
                     └─ DFL → Distribution Focal Loss
```

**与传统 Head 的区别**:
- 传统: 分类和回归共享卷积
- Decoupled: 分类和回归独立分支，互不干扰

**Anchor-Free**:
- 不需要预定义 Anchor Box
- 直接预测目标中心点和宽高
- 更灵活，泛化性更好

---

## 3. 损失函数

### 3.1 总损失

```
Loss = λ_box · L_box + λ_cls · L_cls + λ_dfl · L_dfl
     = 7.5 · L_box + 0.5 · L_cls + 1.5 · L_dfl
```

### 3.2 分类损失 — BCE Loss

```
L_cls = -[y·log(p) + (1-y)·log(1-p)]
```

- 二元交叉熵，逐类别计算
- 使用 Binary Cross Entropy 而非 Softmax

### 3.3 回归损失 — CIoU Loss

```
L_box = 1 - CIoU
CIoU = IoU - (ρ²(b,b_gt)/c²) - α·v
```

- IoU: 交并比
- ρ²: 中心点距离
- v: 宽高比一致性
- 比 GIoU/DIoU 收敛更快

### 3.4 DFL — Distribution Focal Loss

```
L_dfl = -((y_{i+1} - y)·log(S_i) + (y - y_i)·log(S_{i+1}))
```

- 将边界框坐标建模为分布
- 比直接回归更精确
- 提升小目标定位精度

---

## 4. 训练流程

### 4.1 数据增强

| 增强方法 | 概率 | 作用 |
|---------|------|------|
| Mosaic | 1.0 | 4 张图拼接，增加上下文 |
| MixUp | 0.0 | 两张图混合 |
| 随机翻转 | 0.5 | 水平镜像 |
| 随机旋转 | ±10° | 旋转不变性 |
| HSV 增强 | 各种 | 颜色抖动 |
| 随机缩放 | 0.5-1.5 | 尺度不变性 |
| 随机擦除 | 0.4 | 遮挡鲁棒性 |

### 4.2 学习率调度

```
lr
 │
 │    ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │   ╱                        ╲
 │  ╱                          ╲
 │ ╱                            ╲
 │╱                              ╲
 └──────────────────────────────── → epoch
   warmup(3)    cosine decay
```

- Warmup: 前 3 个 epoch 线性增加学习率
- Cosine Decay: 余弦退火，学习率逐渐降低

### 4.3 训练技巧

1. **预训练权重**: 使用 COCO 预训练的权重，fine-tune 效果更好
2. **混合精度**: AMP (FP16) 加速训练，节省显存
3. **早停**: patience=100，防止过拟合
4. **关闭 Mosaic**: 最后 10 个 epoch 关闭 Mosaic 增强

---

## 5. 评估指标

### 5.1 IoU (交并比)

```
IoU = |A ∩ B| / |A ∪ B|
```

### 5.2 Precision & Recall

```
Precision = TP / (TP + FP)  — 查准率
Recall = TP / (TP + FN)    — 查全率
```

### 5.3 AP & mAP

```
AP = ∫ Precision(Recall) dRecall   — PR 曲线下面积
mAP = mean(AP_class)               — 各类 AP 均值
```

- **mAP@0.5**: IoU 阈值 = 0.5 时的 mAP
- **mAP@0.5:0.95**: IoU 阈值从 0.5 到 0.95 的平均 mAP

---

## 6. VisDrone 数据集特点

### 6.1 数据集概况

- 来源: 天津大学无人机航拍
- 场景: 城市道路、校园、商业区
- 高度: 无人机视角 (俯视)
- 挑战: 小目标密集、遮挡严重、尺度变化大

### 6.2 类别分布

| 类别 | 占比 | 检测难度 |
|------|------|---------|
| car | ~35% | 中 |
| pedestrian | ~25% | 高 (小目标) |
| people | ~15% | 高 (小目标) |
| truck | ~8% | 中 |
| bus | ~5% | 低 |
| motor | ~5% | 高 |
| bicycle | ~4% | 高 |
| van | ~2% | 中 |
| tricycle | ~1% | 高 |
| awning-tricycle | ~0.5% | 高 |

### 6.3 小目标挑战

- 大部分目标 < 32×32 像素 (在原始 2000×1500 图片中)
- 缩放到 640×640 后，目标仅 ~10×10 像素
- 需要高分辨率特征图 (P2 层) 来检测

---

## 7. 改进方向

### 7.1 网络结构改进

1. **增加 P2 检测头**: 160×160 特征图，专门检测小目标
2. **注意力机制**: CBAM/SE/ECA，聚焦重要特征
3. **BiFPN**: 加权特征融合，比 PANet 更有效
4. **Dynamic Head**: 动态卷积处理多尺度目标

### 7.2 数据层面改进

1. **SAHI 切片推理**: 将大图切成小块分别检测
2. **更强数据增强**: MixUp、CopyPaste、Mosaic9
3. **自适应锚框**: 根据 VisDrone 数据集调整锚框尺寸

### 7.3 训练策略改进

1. **更大输入尺寸**: 640 → 800/1024
2. **更多训练轮数**: 50 → 100+
3. **学习率调优**: 余弦退火 + warmup
4. **类别平衡**: 对稀有类别加权

---

## 8. 参考资源

- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com)
- [YOLO 论文合集](https://github.com/hysts/yolo-papers)
- [VisDrone 数据集](https://github.com/VisDrone/VisDrone-Dataset)
- [CBAM 论文](https://arxiv.org/abs/1807.06521)
