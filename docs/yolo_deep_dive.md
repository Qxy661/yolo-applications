# YOLO 保姆级详解

> 从零理解目标检测到深入 YOLOv8 每一个细节
> 面向: 想真正理解 YOLO 的学习者，而非仅仅跑通代码

---

## 目录

1. [目标检测基础](#1-目标检测基础)
2. [YOLO 系列演进](#2-yolo-系列演进)
3. [YOLOv8 架构深度解析](#3-yolov8-架构深度解析)
4. [损失函数数学推导](#4-损失函数数学推导)
5. [训练流程全链路](#5-训练流程全链路)
6. [推理流程全链路](#6-推理流程全链路)
7. [COCO 评估体系](#7-coco-评估体系)
8. [VisDrone 小目标检测挑战](#8-visdrone-小目标检测挑战)
9. [SAHI 切片推理深度解析](#9-sahi-切片推理深度解析)
10. [关键代码走读](#10-关键代码走读)

---

## 1. 目标检测基础

### 1.1 什么是目标检测？

目标检测 = **定位** + **分类**

- **输入**: 一张图片 (H×W×3)
- **输出**: 若干个检测框，每个框包含:
  - 边界框坐标 `(x, y, w, h)`
  - 类别标签 `class`
  - 置信度分数 `confidence`

与图像分类的区别:

| 任务 | 输出 | 示例 |
|------|------|------|
| 图像分类 | 一个标签 | "这是一张猫的图" |
| 目标检测 | 多个框+标签 | "图中 (100,200) 处有只猫" |
| 语义分割 | 逐像素标签 | "这些像素属于猫" |
| 实例分割 | 逐像素+实例ID | "这里有两只猫，分别在..." |

### 1.2 One-Stage vs Two-Stage

**Two-Stage (两阶段)**: 先生成候选区域，再分类

```
图片 → Region Proposal → RoI Pooling → 分类+回归
         (RPN/Selective Search)
```

代表模型:
- **R-CNN** (2014): Selective Search + CNN，每个候选区域独立提取特征
- **Fast R-CNN** (2015): 共享特征图，RoI Pooling
- **Faster R-CNN** (2016): 用 RPN 替代 Selective Search，端到端训练
- **Cascade R-CNN** (2018): 多阶段级联，逐步提高 IoU 阈值

特点: 精度高，速度慢 (5-20 FPS)

**One-Stage (单阶段)**: 直接预测，无候选区域

```
图片 → CNN 特征提取 → 直接输出检测框+类别
```

代表模型:
- **YOLO** (2016): 将检测视为回归问题
- **SSD** (2016): 多尺度特征图检测
- **RetinaNet** (2017): Focal Loss 解决类别不平衡

特点: 速度快 (30-100 FPS)，精度略低

**为什么 YOLO 能赢？**

早期 one-stage 精度不如 two-stage，原因在于**正负样本极度不平衡**——大量背景区域被当作负样本，淹没少量正样本的梯度。RetinaNet 的 Focal Loss 和 YOLOv5+ 的 SimOTA/TaskAlignedAssigner 解决了这个问题，使 one-stage 精度追平甚至超越 two-stage。

### 1.3 Anchor-Based vs Anchor-Free

**Anchor-Based (基于锚框)**:

- 预定义一组锚框 (anchor boxes)，如 `[8×8, 16×16, 32×32]`
- 模型预测相对于锚框的**偏移量** `(Δx, Δy, Δw, Δh)`
- 需要手动设计锚框尺寸，依赖数据集统计

```
锚框 (16×16) + 偏移量 (Δx, Δy, Δw, Δh) → 预测框
```

代表: Faster R-CNN, SSD, YOLOv2-v5

**Anchor-Free (无锚框)**:

- 不需要预定义锚框
- 直接预测目标中心点到边界的距离 `(l, t, r, b)` 或中心+宽高
- 更简洁，泛化性更好

```
特征图上的点 → 直接预测 (x_center, y_center, w, h)
```

代表: FCOS, CenterNet, YOLOv8

**YOLOv8 的选择**: Anchor-Free，每个网格点直接预测 4 个距离值 (到左、上、右、下边界的距离) + 用 DFL 将距离建模为分布。

### 1.4 特征金字塔 (Feature Pyramid)

小目标检测的核心问题: **小目标在深层特征图上信息丢失**。

一个 640×640 的输入经过 5 次下采样:
- P1: 320×320
- P2: 160×160
- P3: 80×80 ← 小目标特征
- P4: 40×40 ← 中目标特征
- P5: 20×20 ← 大目标特征

P5 的一个像素对应原图 32×32 的区域，一个 10×10 的小目标在 P5 上几乎不存在。

**FPN (Feature Pyramid Network, 2017)**:

```
P5 ──→ 上采样 ──→ + P4 ──→ 上采样 ──→ + P3
                                    (自顶向下)
```

将高层的语义信息传递给低层，使低层特征图也有强语义。

**PANet (Path Aggregation Network, 2018)**:

在 FPN 基础上增加**自底向上**路径:

```
P3 ──→ 下采样 ──→ + P4 ──→ 下采样 ──→ + P5
                   ↑                      ↑
P5 → 上采样 → + P4 → 上采样 → + P3     (FPN)

双方向融合 → 每个尺度都有丰富的位置+语义信息
```

YOLOv8 使用 PANet + FPN 的组合，P3/P4/P5 三个尺度输出。

---

## 2. YOLO 系列演进

### 2.1 YOLOv1 (2016) — 开山之作

**论文**: "You Only Look Once: Unified, Real-Time Object Detection"

**核心思想**: 将检测问题转化为**回归问题**。

```
输入图片 → CNN → S×S 网格 → 每个格子预测 B 个框 + C 个类别概率
7×7 网格，每个格子预测 2 个框，共 98 个候选框
```

**架构**:
- 24 层卷积 + 2 层全连接
- 输入 448×448，输出 7×7×30 (2 个框×5 + 20 个类别)
- 最后用全连接层直接回归坐标

**局限**:
- 每个格子只能预测 2 个物体，密集场景漏检
- 对小目标和群体目标效果差
- 定位精度不高

### 2.2 YOLOv2 (2017) — 更快更强

**论文**: "YOLO9000: Better, Faster, Stronger"

**核心改进**:
1. **Batch Normalization**: 每个卷积后加 BN，收敛更快
2. **High Resolution Classifier**: 分类器预训练用 448×448 (v1 用 224×224)
3. **Anchor Box**: 引入锚框，预测偏移量而非直接坐标
4. **Dimension Cluster**: K-Means 聚类确定锚框尺寸
5. **Fine-Grained Features**: Passthrough 层拼接高低层特征
6. **Multi-Scale Training**: 训练时随机切换输入尺寸 (320-608)

### 2.3 YOLOv3 (2018) — 多尺度检测

**核心改进**:
1. **Darknet-53**: 53 层卷积骨干网络，引入残差连接
2. **多尺度预测**: 3 个尺度 (13×13, 26×26, 52×52)，分别检测大/中/小目标
3. **FPN**: 特征金字塔网络，自顶向下特征融合
4. **9 个锚框**: 3 个尺度 × 3 个锚框/尺度

```
大目标 ← 13×13 特征图
中目标 ← 26×26 特征图
小目标 ← 52×52 特征图  ← 首次对小目标有专门检测
```

### 2.4 YOLOv4 (2020) — 集大成

**核心改进**:
1. **CSPDarknet**: Cross Stage Partial 连接，减少计算量
2. **SPP**: 空间金字塔池化，增大感受野
3. **PANet**: 自底向上特征融合
4. **Mosaic 数据增强**: 4 张图拼接
5. **CIoU Loss**: 考虑中心距离+宽高比的 IoU 变体
6. **DropBlock 正则化**

### 2.5 YOLOv5 (2020) — 工程化标杆

**核心改进** (Ultralytics 工程化):
1. **Focus 层**: 将像素交叉排列到通道维度，减少计算
2. **自适应锚框**: 根据数据集自动计算锚框尺寸
3. **PyTorch 原生实现**: 易于部署和定制
4. **完整的训练/推理/导出 Pipeline**

YOLOv5 不是学术论文，而是 Ultralytics 的开源项目，但因其工程化质量极高，成为工业界最广泛使用的版本。

### 2.6 YOLOv8 (2023) — Anchor-Free 时代

**核心改进**:
1. **Anchor-Free**: 去掉锚框，直接预测
2. **C2f 模块**: 替代 C3，梯度流更丰富
3. **Decoupled Head**: 分类和回归独立分支
4. **Task-Aligned Assigner**: 动态正负样本分配
5. **DFL Loss**: 分布式焦点损失，更精确的定位

```
YOLOv1 → v2 → v3 → v4 → v5 → v8
 │       │      │      │      │      │
直觉级   锚框   多尺度  集大成  工程化  Anchor-Free
```

---

## 3. YOLOv8 架构深度解析

### 3.1 整体架构图

```
输入 (1, 3, 640, 640)
    │
    ▼
┌─ Backbone (CSPDarknet) ──────────────────────────┐
│  Conv(3→16, 3×3, s=2) → 320×320×16              │
│  Conv(16→32, 3×3, s=2) → 160×160×32             │
│  C2f(32→32, n=1) → 160×160×32       ← P2       │
│  Conv(32→64, 3×3, s=2) → 80×80×64               │
│  C2f(64→64, n=2) → 80×80×64          ← P3 输出  │
│  Conv(64→128, 3×3, s=2) → 40×40×128             │
│  C2f(128→128, n=2) → 40×40×128       ← P4 输出  │
│  Conv(128→256, 3×3, s=2) → 20×20×256            │
│  C2f(256→256, n=1) → 20×20×256       ← P5       │
│  SPPF(256→256) → 20×20×256                       │
└──────────────────────────────────────────────────┘
    │                    │                    │
    P3 (80×80×64)        P4 (40×40×128)       P5 (20×20×256)
    │                    │                    │
    ▼                    ▼                    ▼
┌─ Neck (PANet + FPN) ────────────────────────────┐
│  FPN (自顶向下):                                  │
│    P5 → Upsample → Concat(P4) → C2f → P4_out   │
│    P4_out → Upsample → Concat(P3) → C2f → P3_out│
│  PAN (自底向上):                                  │
│    P3_out → Conv(s=2) → Concat(P4_out) → C2f    │
│    P4_out → Conv(s=2) → Concat(P5) → C2f        │
└──────────────────────────────────────────────────┘
    │                    │                    │
    P3_out (80×80)       P4_out (40×40)       P5_out (20×20)
    │                    │                    │
    ▼                    ▼                    ▼
┌─ Head (Decoupled, Anchor-Free) ─────────────────┐
│  每个尺度:                                       │
│    分类分支: Conv → Conv → cls_pred (nc)         │
│    回归分支: Conv → Conv → box_pred (4×reg_max)  │
│                                                  │
│  检测点数: 80²+40²+20² = 8400 个点               │
│  每个点预测: nc 个类别 + 4×reg_max 个距离值       │
└──────────────────────────────────────────────────┘
    │
    ▼
后处理: DFL 解码 → NMS → 最终检测结果
```

### 3.2 核心模块详解

#### 3.2.1 Conv 模块 (基础卷积)

```python
# 标准 Conv-BN-SiLU 三件套
Conv(in_ch, out_ch, kernel=3, stride=2, padding=1)
  → Conv2d → BatchNorm2d → SiLU 激活
```

- **SiLU (Swish)**: `x × sigmoid(x)`，比 ReLU 更平滑，梯度流更好
- **BN**: 加速收敛，起到正则化作用

#### 3.2.2 C2f 模块 (核心骨干模块)

C2f 是 YOLOv8 的核心构建块，替代了 YOLOv5 的 C3:

```
输入 x
  │
  ├─→ Conv(1×1) → split → [x1, x2]
  │                        │
  │                        x1 → Bottleneck → y1
  │                        x2 → Bottleneck → y2
  │                        y1 → Bottleneck → y3
  │                        ...
  │
  └─→ Concat([x1, x2, y1, y2, y3, ...]) → Conv(1×1) → 输出
```

**与 C3 的区别**:
- C3: 拆分 → N 个 Bottleneck → 拼接 (只拼接最后两个)
- C2f: 拆分 → N 个 Bottleneck → **全部拼接** (保留所有中间梯度)

全部拼接的好处: 梯度流更丰富，训练更稳定，参数量几乎不变。

#### 3.2.3 Bottleneck 模块

```python
# 残差块结构
Bottleneck(in_ch, out_ch):
  identity = x
  x = Conv(1×1)  # 降维
  x = Conv(3×3)  # 深度卷积
  x = x + identity  # 残差连接
```

#### 3.2.4 SPPF (空间金字塔池化 - 快速版)

```
输入 → Conv(1×1) → MaxPool(5) → MaxPool(5) → MaxPool(5)
         │            │            │            │
         └────────────┴────────────┴────────────┘
                         Concat → Conv(1×1) → 输出
```

**作用**: 用不同尺度的池化核 (5×5, 9×9, 13×13) 提取多尺度特征，增大感受野。
**SPPF vs SPP**: SPPF 串联 3 次 5×5 MaxPool = 并联 5×5 + 9×9 + 13×13，计算量更少。

#### 3.2.5 上采样 (Upsample)

```python
Upsample(scale_factor=2, mode='nearest')
```

将特征图尺寸放大 2 倍，用最近邻插值。在 Neck 中用于 FPN 的自顶向下路径。

#### 3.2.6 Decoupled Head

```
特征图 (B, C, H, W)
  │
  ├─→ cls_conv(1×1) → cls_conv(1×1) → cls_pred (B, nc, H, W)
  │
  └─→ reg_conv(1×1) → reg_conv(1×1) → reg_pred (B, 4×reg_max, H, W)
```

- **分类分支**: 输出 `nc` 个类别概率 (多标签，用 sigmoid 而非 softmax)
- **回归分支**: 输出 `4 × reg_max` 个值，用 DFL 解码为 4 个距离

**Decoupled vs Coupled**:
- Coupled (YOLOv5): 分类和回归共享卷积层，互相干扰
- Decoupled (YOLOv8): 各自独立，分类不影响回归，回归不影响分类

### 3.3 参数量与计算量

| 模型 | 参数量 | GFLOPs | mAP (COCO) |
|------|--------|--------|------------|
| YOLOv8n | 3.2M | 8.7 | 37.3 |
| YOLOv8s | 11.2M | 28.6 | 44.9 |
| YOLOv8m | 25.9M | 78.9 | 50.2 |
| YOLOv8l | 43.7M | 165.2 | 52.9 |
| YOLOv8x | 68.2M | 257.8 | 53.9 |

**本项目选择**:
- 基线: YOLOv8n (3.2M) — 轻量，快速验证
- 改进: YOLOv8s (11.2M) — 更强表达能力，速度仍可接受

---

## 4. 损失函数数学推导

### 4.1 总损失公式

```
L_total = λ_box · L_box + λ_cls · L_cls + λ_dfl · L_dfl
        = 7.5   · L_box + 0.5   · L_cls + 1.5   · L_dfl
```

权重设计理由:
- **box=7.5**: 定位精度对检测至关重要，给予最高权重
- **cls=0.5**: 分类相对容易，权重较低
- **dfl=1.5**: DFL 辅助定位，中等权重

### 4.2 IoU 家族

#### 4.2.1 IoU (Intersection over Union)

```
IoU = |A ∩ B| / |A ∪ B| = 交集面积 / 并集面积
```

取值范围 [0, 1]，1 表示完全重叠。

**问题**: 当两个框不相交时，IoU = 0，无法提供梯度方向。

#### 4.2.2 GIoU (Generalized IoU)

```
GIoU = IoU - |C \ (A ∪ B)| / |C|
```

其中 C 是 A 和 B 的最小外接矩形。

**改进**: 即使不相交，GIoU 也能提供梯度。但当一个框完全包含另一个框时，GIoU 退化为 IoU。

#### 4.2.3 DIoU (Distance IoU)

```
DIoU = IoU - ρ²(b, b_gt) / c²
```

其中:
- `b, b_gt`: 预测框和真实框的中心点
- `ρ²`: 两点间的欧氏距离平方
- `c`: 最小外接矩形的对角线长度

**改进**: 直接优化中心点距离，收敛更快。

#### 4.2.4 CIoU (Complete IoU) — YOLOv8 使用

```
CIoU = IoU - ρ²(b, b_gt) / c² - α · v

其中:
  v = (4/π²) × (arctan(w_gt/h_gt) - arctan(w/h))²
  α = v / (1 - IoU + v)
```

**三重优化**:
1. **IoU**: 重叠面积
2. **中心距离**: ρ²/c² 惩罚中心偏移
3. **宽高比一致性**: α·v 惩罚形状差异

```
L_box = 1 - CIoU
```

### 4.3 分类损失 — BCE Loss

YOLOv8 使用**多标签分类** (sigmoid)，而非互斥分类 (softmax):

```
L_cls = -1/N × Σ [y_i × log(σ(p_i)) + (1 - y_i) × log(1 - σ(p_i))]
```

其中:
- `y_i ∈ {0, 1}`: 第 i 个类别的标签
- `σ(p_i) = 1/(1+exp(-p_i))`: sigmoid 激活
- `N`: 正样本数量

**为什么用 sigmoid 而非 softmax?**
- 一个物体可能同时属于多个类别 (如 "红色汽车" 既是 "红色" 又是 "汽车")
- VisDrone 中 "pedestrian" 和 "people" 存在语义重叠
- sigmoid 逐类别独立计算，允许同时属于多个类别

### 4.4 DFL — Distribution Focal Loss

**核心思想**: 不直接回归一个精确坐标值，而是将其建模为**离散分布**。

假设 `reg_max = 16`，表示将距离 [0, 16] 离散化为 17 个点 {0, 1, 2, ..., 16}。

对于真实距离值 `y = 7.3`:
- 分布: `[0, 0, 0, 0, 0, 0, 0, 0.7, 0.3, 0, 0, 0, 0, 0, 0, 0, 0]`
- `y_i = 7`, `y_{i+1} = 8`, `y - y_i = 0.3`, `y_{i+1} - y = 0.7`

```
L_dfl = -((y_{i+1} - y) × log(S_i) + (y - y_i) × log(S_{i+1}))
```

其中 `S_i`, `S_{i+1}` 是模型预测的概率分布中对应位置的值。

**为什么用 DFL 而非直接回归?**
- 直接回归 `w = f(x)` 是一个确定性输出，无法表达不确定性
- DFL 将坐标建模为分布，模型可以表达 "我确定在 7 到 8 之间，更偏向 7"
- 对小目标特别有效，因为小目标的 1 像素误差影响巨大，需要更精确的定位

**解码过程**:
```
模型输出: 4 × reg_max 个 logits
    ↓ softmax
概率分布: 4 × 17 个概率值
    ↓ 加权求和
距离值: dist = Σ(i × S_i), i = 0..16
    ↓ 转换
中心点: x_c = (dist_left - dist_right + 2×anchor_x) / 2
        y_c = (dist_top - dist_bottom + 2×anchor_y) / 2
宽高:   w = dist_left + dist_right
        h = dist_top + dist_bottom
```

---

## 5. 训练流程全链路

### 5.1 数据增强 Pipeline

YOLOv8 训练时的数据增强分为**离线增强**和**在线增强**:

#### 在线增强 (训练时实时应用):

```
原始图片 + 标注
    │
    ▼
┌─ Mosaic (1.0) ──────────────────────┐
│  随机选 3 张图 + 当前图，拼成 2×2    │
│  作用: 增加上下文，模拟多尺度        │
│  注意: 最后 10 个 epoch 关闭         │
└────────────────────────────────────┘
    │
    ▼
┌─ MixUp (0.15) ─────────────────────┐
│  两张图按比例混合                    │
│  混合系数 β ~ Beta(8, 8)            │
│  作用: 增加数据多样性               │
└────────────────────────────────────┘
    │
    ▼
┌─ 随机仿射变换 ──────────────────────┐
│  平移: ±0.1                         │
│  缩放: 0.5 ~ 1.5                    │
│  旋转: ±10°                         │
│  剪切: ±0.0                         │
└────────────────────────────────────┘
    │
    ▼
┌─ 颜色增强 (HSV) ────────────────────┐
│  H (色调): ±0.015                   │
│  S (饱和度): ±0.7                   │
│  V (明度): ±0.4                     │
└────────────────────────────────────┘
    │
    ▼
┌─ 随机翻转 ─────────────────────────┐
│  水平翻转概率: 0.5                  │
│  垂直翻转: 关闭                     │
└────────────────────────────────────┘
    │
    ▼
┌─ 随机擦除 (0.4) ───────────────────┐
│  随机遮挡一个区域                   │
│  作用: 提高遮挡鲁棒性               │
└────────────────────────────────────┘
    │
    ▼
  Letterbox (缩放到 640×640, 灰色填充)
    │
    ▼
  归一化 (像素值 / 255.0)
```

**Mosaic 增强的细节**:

```
┌──────┬──────┐
│  图1 │  图2 │
│      │      │
├──────┼──────┤
│  图3 │  图4 │ ← 当前图 (随机位置)
│      │      │
└──────┴──────┘

拼接点: 随机选择 (cx, cy) 在 [0.25W, 0.75W] × [0.25H, 0.75H]
```

Mosaic 的好处:
- 一次训练看到 4 张图，相当于 batch 扩大 4 倍
- 拼接处的物体被部分截断，模拟遮挡
- 不同大小的物体混合，模拟多尺度

#### 本项目的增强策略

| 增强方法 | 基线 (YOLOv8n) | 改进 (YOLOv8s) |
|---------|---------------|---------------|
| Mosaic | 1.0 | 1.0 |
| MixUp | 0.0 | **0.15** |
| CopyPaste | 0.0 | **0.1** |
| 随机擦除 | 0.4 | **0.5** |
| 旋转 | ±10° | **±20°** |
| 缩放 | 0.5-1.5 | **0.3-1.7** |

改进版使用了更强的增强策略，增加数据多样性，防止过拟合。

### 5.2 学习率调度

```
lr
 │
 │      ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │     ╱                          ╲
 │    ╱                            ╲
 │   ╱                              ╲
 │  ╱                                ╲
 │ ╱                                  ╲
 │╱                                    ╲
 └────────────────────────────────────── → epoch
   3 epochs        77 epochs
   (warmup)        (cosine decay)
```

**Warmup 阶段** (前 3 个 epoch):
```
lr(epoch) = lr0 × (epoch / warmup_epochs)
         = 0.01 × (epoch / 3)
```
从 0 线性增长到 0.01。原因: 训练初期权重随机，大学习率会导致梯度爆炸。

**Cosine Decay 阶段** (第 4-80 个 epoch):
```
lr(epoch) = lrf × lr0 × (1 + cos(π × progress)) / 2
         = 0.01 × 0.01 × (1 + cos(π × progress)) / 2
```
余弦衰减比线性衰减更平滑，避免学习率突变。

### 5.3 正负样本分配

**传统方法 (YOLOv3-v5)**: IoU-based 匹配
- 预设 anchor 与 GT 的 IoU > 阈值 → 正样本
- 固定阈值，不够灵活

**YOLOv8 的 Task-Aligned Assigner**:

同时考虑**分类质量**和**定位质量**:

```
alignment_metric = s^α × t^β

s = 分类分数 (confidence)
t = IoU (预测框与GT)
α, β = 可调参数
```

步骤:
1. 计算所有预测框与所有 GT 的 alignment_metric
2. 对每个 GT，选择 alignment_metric 最高的前 k 个预测作为正样本
3. 动态调整，不依赖固定阈值

**与 SimOTA (YOLOX) 的区别**:
- SimOTA: 基于代价矩阵的最优传输
- TaskAligned: 基于质量指标的 top-k 选择，更简单高效

### 5.4 优化器

YOLOv8 使用 **SGD** (默认) 或 **AdamW**:

```python
# SGD (默认)
optimizer = SGD(lr=0.01, momentum=0.937, weight_decay=0.0005)

# AdamW (可选)
optimizer = AdamW(lr=0.01, weight_decay=0.0005)
```

**动量 (Momentum)**:
```
v_t = momentum × v_{t-1} + gradient
w_t = w_{t-1} - lr × v_t
```
动量 = 0.937 表示 93.7% 的历史梯度惯性，帮助穿越局部最优。

---

## 6. 推理流程全链路

### 6.1 完整推理流程

```
输入图片 (H, W, 3)
    │
    ▼
┌─ 前处理 ───────────────────────────┐
│  1. 读取图片 (BGR → RGB)           │
│  2. Letterbox 缩放到 640×640       │
│  3. 归一化 (÷255)                  │
│  4. HWC → CHW 转置                 │
│  5. 添加 batch 维度 → (1, 3, 640, 640)│
└────────────────────────────────────┘
    │
    ▼
┌─ 模型推理 ─────────────────────────┐
│  Backbone → Neck → Head            │
│  输出:                              │
│    cls_pred: (1, nc, 8400)          │
│    reg_pred: (1, 4×reg_max, 8400)  │
└────────────────────────────────────┘
    │
    ▼
┌─ 后处理 ───────────────────────────┐
│  1. DFL 解码 → 每个点的 (x1,y1,x2,y2)│
│  2. 置信度过滤 (conf > 0.25)       │
│  3. NMS (IoU > 0.45 的框去重)      │
│  4. 坐标映射回原图尺寸              │
└────────────────────────────────────┘
    │
    ▼
检测结果: [(x1,y1,x2,y2, conf, class), ...]
```

### 6.2 Letterbox 缩放

将任意尺寸图片缩放到正方形，同时保持宽高比:

```
原图 (1920×1080)
    │
    ▼ 计算缩放比例
r = min(640/1920, 640/1080) = 0.333
新尺寸: 640 × 360
    │
    ▼ 填充灰色
画布: 640×640，灰色填充
图片放置在: y_offset = (640-360)/2 = 140
最终: 灰色带(上) + 图片(640×360) + 灰色带(下)
```

### 6.3 DFL 解码

```python
# 输入: reg_pred (1, 4×16, 8400)
# 1. Reshape → (1, 4, 16, 8400)
# 2. Softmax → 概率分布
# 3. 加权求和 → 距离值

dist = Σ(i × softmax(reg_pred)[i]), i = 0..15
# dist_left, dist_top, dist_right, dist_bottom

# 4. 转换为坐标
x_center = (anchor_x + dist_right - dist_left) / 2
y_center = (anchor_y + dist_bottom - dist_top) / 2
width = dist_left + dist_right
height = dist_top + dist_bottom

# 5. 转换为 (x1, y1, x2, y2)
x1 = x_center - width/2
y1 = y_center - height/2
x2 = x_center + width/2
y2 = y_center + height/2
```

### 6.4 NMS (Non-Maximum Suppression)

NMS 是去除冗余检测框的标准后处理:

```
输入: 所有预测框 + 置信度 + 类别
    │
    ▼ 按置信度降序排列
    │
    ▼ 循环:
    │  取置信度最高的框 → 加入结果
    │  删除与该框 IoU > 0.45 的同类框
    │  重复直到没有框剩余
    │
    ▼
输出: 去重后的检测框
```

**为什么需要 NMS?**
- 一个物体可能被多个检测点预测到
- NMS 保留置信度最高的，去除重叠的

**YOLOv8 的 NMS 设置**:
- `conf_threshold = 0.25`: 置信度 < 0.25 的框直接丢弃
- `iou_threshold = 0.45`: IoU > 0.45 的框视为重复
- `max_det = 300`: 每张图最多保留 300 个框
- `class_agnostic = False`: NMS 按类别独立进行

---

## 7. COCO 评估体系

### 7.1 IoU 阈值

**mAP@0.5**: IoU 阈值固定为 0.5
- 预测框与 GT 的 IoU ≥ 0.5 → TP
- 宽松标准，容忍定位误差

**mAP@0.5:0.95**: IoU 阈值从 0.5 到 0.95，步长 0.05
```
mAP@0.5:0.95 = mean(mAP@0.5, mAP@0.55, mAP@0.6, ..., mAP@0.95)
```
- 10 个 IoU 阈值的平均值
- 严格标准，要求精确定位

### 7.2 Precision-Recall 曲线

对于每个类别，按置信度降序排列所有预测:

```
阈值从高到低:
  高阈值 → 少量高置信预测 → Precision 高, Recall 低
  低阈值 → 大量预测 → Precision 低, Recall 高
```

**绘制 PR 曲线**:
- X 轴: Recall (0 → 1)
- Y 轴: Precision
- 曲线通常先高后低 (高置信度预测准确率高)

### 7.3 AP 计算

**COCO 的 AP 计算** (11 点插值法):

```
AP = (1/11) × Σ max_{r' ≥ r} P(r')
     r ∈ {0, 0.1, 0.2, ..., 1.0}
```

即在 11 个 Recall 点上，取每个点右侧最高的 Precision 值，求平均。

**更精确的方式** (所有点插值):

```
AP = Σ (r_{n+1} - r_n) × max_{r' ≥ r_n} P(r')
```

### 7.4 mAP 计算

```
mAP = (1/C) × Σ AP_c
     c = 1..C
```

C 是类别数，对所有类别的 AP 取平均。

**本项目的 mAP**:
```
mAP@0.5 = mean(AP_pedestrian, AP_people, ..., AP_motor)
        = mean(0.479, 0.351, 0.176, 0.815, 0.478, 0.405, 0.308, 0.171, 0.589, 0.487)
        = 0.4258
```

### 7.5 完整评估流程

```
模型权重 (best.pt)
    │
    ▼ 在验证集上推理
每个图片 → 模型 → 预测框列表
    │
    ▼ 匹配
对每个 GT，找 IoU 最高的预测框
  IoU ≥ 0.5 → TP (True Positive)
  无匹配 → FN (False Negative)
  IoU < 0.5 → FP (False Positive)
    │
    ▼ 计算指标
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
AP = PR 曲线下面积
mAP = 所有类别 AP 的均值
```

---

## 8. VisDrone 小目标检测挑战

### 8.1 什么是小目标？

COCO 的定义:
| 尺寸 | 像素面积 | 示例 |
|------|---------|------|
| Small | < 32² = 1024 | 远处的行人 |
| Medium | 32² ~ 96² | 骑车的人 |
| Large | > 96² | 近处的汽车 |

VisDrone 中的尺度分布:
- **Small (< 32²)**: ~60% 的目标
- **Medium (32²~96²)**: ~30%
- **Large (> 96²)**: ~10%

### 8.2 小目标为什么难检测？

1. **特征信息少**: 10×10 的目标在 640×640 输入中只占 0.02% 的像素
2. **下采样丢失**: 经过 5 次 stride=2 的卷积，10×10 目标在 P5 上不存在
3. **标注噪声**: 小目标的标注框偏差 2 像素就可能 IoU < 0.5
4. **语义不足**: 浅层特征图 (P2/P3) 位置信息好但语义弱

### 8.3 本项目的应对策略

| 策略 | 原理 | 效果 |
|------|------|------|
| 更大输入 (800px) | 保留更多像素信息 | mAP +15% |
| 更强模型 (v8s) | 更多参数表达细微特征 | mAP +20% |
| MixUp/CopyPaste | 增加小目标样本多样性 | mAP +5% |
| SAHI 切片推理 | 小目标在切片中像素占比增大 | mAP +6% |
| 降低置信度阈值 | 保留更多低置信度的真实检测 | mAP +8% |

### 8.4 SAHI 对小目标的提升原理

```
原图 1360×765: 行人约 15×30 像素 (占原图 0.08%)
    │
    ▼ 切片为 640×640 (重叠 20%)
每个切片: 行人约 15×30 像素 (占切片 0.11%)
    │ 但切片分辨率更高，目标在切片中的相对尺寸更大
    ▼
检测效果提升: bicycle +44.9%, pedestrian +16.3%
```

关键: SAHI 不是改变目标像素数，而是改变**输入-目标的比例关系**。在 640×640 的切片中，模型看到的 "视野" 更小，小目标的相对尺寸更大，更容易被检测。

---

## 9. SAHI 切片推理深度解析

### 9.1 SAHI 工作原理

```
输入图片 (H, W)
    │
    ▼ 计算切片坐标
slice_positions = get_slice_positions(
    image_size=(H, W),
    slice_size=(640, 640),
    overlap_ratio=0.2
)
    │
    ▼ 逐切片推理
for (y, x) in slice_positions:
    slice = image[y:y+640, x:x+640]
    predictions = model.predict(slice, conf=0.01)  # 低阈值
    # 坐标映射回原图
    for pred in predictions:
        pred.x += x
        pred.y += y
    all_predictions.extend(predictions)
    │
    ▼ NMS 合并
final_predictions = nms(all_predictions, iou_threshold=0.5)
```

### 9.2 切片坐标计算

```
原图: 1360 × 765
切片: 640 × 640
重叠: 0.2 (128 像素)
步长: 640 × (1 - 0.2) = 512

X 方向: 0, 512, 1024  → 3 个切片
Y 方向: 0, 512         → 2 个切片
总切片: 3 × 2 = 6 个

┌─────┬─────┬─────┐
│ (0,0)│(512,0)│(1024,0)│
│ 640  │ 640  │ 336×640│
├─────┼─────┼─────┤
│(0,512)│(512,512)│(1024,512)│
│ 640  │ 640  │ 336×253│
└─────┴─────┴─────┘
```

注意最后一个切片可能超出图片边界，需要特殊处理 (填充或裁剪)。

### 9.3 NMS 合并策略

不同切片可能检测到同一个物体 (因为重叠区域)，需要用 NMS 去重:

```
切片 1 检测到: 汽车 A (conf=0.9)
切片 2 检测到: 汽车 A (conf=0.85)  ← 重叠区域，同辆车
切片 3 检测到: 汽车 B (conf=0.7)

NMS 处理:
  汽车 A: 保留 conf=0.9 的，删除 conf=0.85 的
  汽车 B: 保留
```

### 9.4 SAHI 关键参数

| 参数 | 值 | 影响 |
|------|-----|------|
| slice_height/width | 640 | 必须匹配模型训练尺寸 |
| overlap_ratio | 0.2 | 太小→边界漏检，太大→计算浪费 |
| conf_threshold | 0.01~0.25 | 越低→召回高但噪声多 |
| postprocess_match_threshold | 0.5 | NMS 的 IoU 阈值 |

### 9.5 SAHI vs 普通推理

| 场景 | 普通推理 | SAHI |
|------|---------|------|
| 大图缩放到 640 | 小目标信息丢失 | 切片保留原始分辨率 |
| 模型输入 | 直接缩放 | 分块处理 |
| 推理时间 | 1x | ~6x (6 个切片) |
| 小目标检测 | 较差 | 显著提升 |

---

## 10. 关键代码走读

### 10.1 训练脚本 (`src/train.py`)

**核心逻辑**:
```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # 加载预训练权重

model.train(
    data="data/visdrone/visdrone.yaml",  # 数据集配置
    epochs=80,
    imgsz=800,
    batch=8,
    # 超参数
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    close_mosaic=10,
    amp=True,
)
```

**参数解读**:
- `optimizer="auto"`: Ultralytics 根据 batch size 自动选择 SGD 或 AdamW
- `close_mosaic=10`: 最后 10 个 epoch 关闭 Mosaic，因为 Mosaic 会引入噪声，后期不需要
- `amp=True`: 混合精度训练，FP16 前向 + FP32 反向，速度更快，显存更省

### 10.2 评估脚本 (`src/evaluate.py`)

```python
model = YOLO(weights)
metrics = model.val(
    data=data_yaml,
    imgsz=640,
    batch=16,
    conf=0.001,   # 评估时不设阈值，保留所有预测
    iou=0.6,      # NMS IoU
    verbose=True,
)

# 访问指标
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")

# 各类别 AP
for i, name in enumerate(metrics.names):
    print(f"{name}: {metrics.box.ap50[i]:.4f}")
```

### 10.3 SAHI 评估 (`src/sahi_eval.py`)

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 加载模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=weights,
    confidence_threshold=0.01,  # 低阈值，后续再过滤
    device="cuda:0",
)

# 切片推理
result = get_sliced_prediction(
    image=image,
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type="NMS",
    postprocess_match_threshold=0.5,
)

# 结果: ObjectPrediction 列表
for pred in result.object_prediction_list:
    bbox = pred.bbox  # BoundingBox
    score = pred.score
    category = pred.category
```

### 10.4 零训练优化 (`src/optimize.py`)

**阈值网格搜索原理**:
```python
# 1. 以极低阈值 (0.01) 收集所有预测
all_predictions = sahi_inference(conf=0.01)

# 2. 按不同阈值过滤，评估
for conf_threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    filtered = [p for p in all_predictions if p.score >= conf_threshold]
    mAP = evaluate(filtered, ground_truth)
    print(f"conf={conf_threshold}: mAP={mAP}")
```

**TTA 原理**:
```python
# 原图推理
pred1 = sahi_inference(image)

# 水平翻转推理
flipped = cv2.flip(image, 1)
pred2 = sahi_inference(flipped)
# 坐标映射: x' = W - x

# 合并后 NMS
all_preds = pred1 + pred2
final = nms(all_preds, iou_threshold=0.5)
```

---

## 参考文献

1. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR, 2016.
2. Redmon, J., Farhadi, A. "YOLO9000: Better, Faster, Stronger." CVPR, 2017.
3. Redmon, J., Farhadi, A. "YOLOv3: An Incremental Improvement." arXiv, 2018.
4. Bochkovskiy, A., et al. "YOLOv4: Optimal Speed and Accuracy." arXiv, 2020.
5. Jocher, G., et al. "Ultralytics YOLOv5." GitHub, 2020.
6. Jocher, G., et al. "Ultralytics YOLOv8." GitHub, 2023.
7. Lin, T.Y., et al. "Feature Pyramid Networks for Object Detection." CVPR, 2017.
8. Liu, S., et al. "Path Aggregation Network." CVPR, 2018.
9. Zheng, Z., et al. "Distance-IoU Loss." AAAI, 2020.
10. Li, X., et al. "Generalized Focal Loss." NeurIPS, 2020.
11. Feng, C., et al. "TOOD: Task-Aligned One-Stage Object Detection." ICCV, 2021.
12. Akhan, F., et al. "SAHI: Slicing Aided Hyper Inference." arXiv, 2022.
