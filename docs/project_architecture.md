# 项目架构全解析 —— 文件结构、数据流与技术实现

> 本文档详细解析项目的每一个文件、目录、配置和脚本之间的关系与作用
> 展示从原始数据到最终检测结果的完整数据流
> 深入讲解每个技术实现背后的 YOLO 知识

---

## 目录

- [一、项目全景图](#一项目全景图)
- [二、目录结构详解](#二目录结构详解)
- [三、数据层：从原始数据到训练数据](#三数据层从原始数据到训练数据)
- [四、模型层：YOLOv8 架构定义与扩展](#四模型层yolov8-架构定义与扩展)
- [五、训练层：三个训练脚本的设计逻辑](#五训练层三个训练脚本的设计逻辑)
- [六、评估层：多维度评估体系](#六评估层多维度评估体系)
- [七、优化层：零训练优化流水线](#七优化层零训练优化流水线)
- [八、可视化层：检测结果的可视化展示](#八可视化层检测结果的可视化展示)
- [九、文档层：完整的知识沉淀体系](#九文档层完整的知识沉淀体系)
- [十、数据流全景图](#十数据流全景图)
- [十一、文件依赖关系矩阵](#十一文件依赖关系矩阵)
- [十二、YOLO 知识在项目中的应用映射](#十二yolo-知识在项目中的应用映射)

---

# 一、项目全景图

## 1.1 项目定位

这是一个**端到端的目标检测项目**，覆盖了从数据准备到模型部署的完整流程：

```
原始数据 → 数据预处理 → 模型训练 → 模型评估 → 推理优化 → 结果展示
   ↓          ↓           ↓          ↓          ↓          ↓
 VisDrone  prepare_data  train.py  evaluate.py optimize.py visualize.py
  .zip      .py          train_improved.py     sahi_eval.py
                          train_p2.py
```

## 1.2 五轮实验与文件对应关系

| 实验 | 训练脚本 | 评估脚本 | 配置文件 | 结果文件 |
|------|---------|---------|---------|---------|
| 第一轮: 基线 | `src/train.py` | `src/evaluate.py` | 默认 yolov8.yaml | `results/baseline_metrics.json` |
| 第二轮: 改进 | `src/train_improved.py` | `src/evaluate.py` | 默认 yolov8.yaml | `results/improved_metrics.json` |
| 第三轮: SAHI | (复用改进模型) | `src/sahi_eval.py` | - | `results/sahi_metrics.json` |
| 第四轮: 零训练 | (复用改进模型) | `src/optimize.py` | - | `results/threshold_search.json` 等 |
| 第五轮: P2+CBAM | `src/train_p2.py` | `src/evaluate_all.py` | `configs/yolov8s-p2.yaml` | (训练中) |

## 1.3 核心设计原则

1. **关注点分离**: 每个脚本只负责一个任务（训练/评估/优化/可视化）
2. **配置驱动**: 模型架构通过 YAML 配置定义，不硬编码
3. **结果可追溯**: 每次实验的结果保存为 JSON，可对比、可复现
4. **渐进式改进**: 五轮实验递进，每轮解决一个特定问题

---

# 二、目录结构详解

## 2.1 完整目录树

```
E:\yolo-visdrone\                          # 项目根目录
│
├── configs/                               # 模型架构配置 (YAML)
│   └── yolov8s-p2.yaml                   # ★ P2+CBAM 自定义模型配置
│
├── data/                                  # 数据集
│   └── visdrone/                          # VisDrone2019-DET
│       ├── visdrone.yaml                  # ★ YOLO 数据集配置文件
│       ├── images/                        # 图片文件
│       │   ├── train/                     # 训练集 (6,471 张)
│       │   └── val/                       # 验证集 (548 张)
│       ├── labels/                        # YOLO 格式标注
│       │   ├── train/                     # 训练集标注
│       │   └── val/                       # 验证集标注
│       └── zips/                          # 原始压缩包
│
├── docs/                                  # 项目文档
│   ├── experiments/                       # 分实验报告
│   │   ├── exp1_baseline.md              # 实验一: 基线
│   │   ├── exp2_improved.md              # 实验二: 改进
│   │   ├── exp3_sahi.md                  # 实验三: SAHI
│   │   └── exp4_zero_training.md         # 实验四: 零训练优化
│   ├── experiment_log.md                  # 实验过程记录
│   ├── experiment_report.md               # 标准化实验报告
│   ├── learning_summary.md                # 学习认知总结
│   ├── technical_report.md                # 技术总结报告
│   ├── paper_review.md                    # 论文调研
│   ├── usage_guide.md                     # 使用教程
│   ├── yolo_deep_dive.md                  # YOLO 保姆级详解
│   ├── yolo_guide.md                      # YOLO 快速入门
│   └── project_architecture.md            # ★ 本文档
│
├── ppt/                                   # 演示文稿
│   └── presentation.md                    # 结题汇报 PPT
│
├── results/                               # 实验结果 (JSON)
│   ├── baseline_metrics.json              # 基线评估结果
│   ├── improved_metrics.json              # 改进评估结果
│   ├── sahi_metrics.json                  # SAHI 评估结果
│   ├── sahi_tta_metrics.json              # SAHI+TTA 评估结果
│   ├── threshold_search.json              # 置信度阈值搜索结果
│   ├── ensemble_metrics.json              # 集成评估结果
│   ├── optimization_summary.json          # 优化实验总结
│   ├── baseline/                          # 基线检测可视化
│   ├── improved/                          # 改进检测可视化
│   ├── baseline_detections/               # 基线检测框图
│   ├── improved_detections/               # 改进检测框图
│   ├── compare/                           # 检测对比图
│   └── sahi_compare/                      # SAHI 对比图
│
├── runs/                                  # Ultralytics 训练输出
│   ├── detect/
│   │   └── runs/
│   │       ├── baseline/                  # 基线训练输出
│   │       │   └── yolov8n_visdrone/
│   │       │       ├── weights/           # best.pt, last.pt
│   │       │       ├── results.png        # 训练曲线
│   │       │       ├── confusion_matrix.png
│   │       │       └── args.yaml
│   │       └── improved/                  # 改进训练输出
│   │           └── yolov8s_visdrone/
│   ├── eval/                              # 评估输出
│   └── p2/                                # P2+CBAM 训练输出
│       └── yolov8s_p2_cbam/
│           ├── weights/
│           └── ...
│
├── src/                                   # 源代码
│   ├── train.py                           # ★ 基线训练脚本
│   ├── train_improved.py                  # ★ 改进训练脚本
│   ├── train_p2.py                        # ★ P2+CBAM 训练脚本
│   ├── evaluate.py                        # ★ 单模型评估脚本
│   ├── evaluate_all.py                    # ★ 标准化全量评估
│   ├── optimize.py                        # ★ 零训练优化脚本
│   ├── sahi_eval.py                       # SAHI 评估脚本
│   ├── sahi_detect.py                     # SAHI 检测可视化
│   ├── detect.py                          # 推理检测脚本
│   ├── visualize.py                       # 可视化脚本
│   ├── utils.py                           # 工具函数
│   ├── register_custom_modules.py         # ★ CBAM 模块注册
│   └── cbam.py                            # CBAM 独立实现
│
├── weights/                               # 预训练权重
│   ├── yolov8n.pt                         # YOLOv8n COCO 预训练
│   └── yolov8s.pt                         # YOLOv8s COCO 预训练
│
├── prepare_data.py                        # 数据集准备脚本
├── download_data.py                       # 数据集下载脚本
├── fast_download.py                       # 快速下载脚本
├── monitor_training.py                    # 训练监控脚本
└── README.md                              # 项目说明
```

## 2.2 目录职责划分

| 目录 | 职责 | 核心文件 |
|------|------|---------|
| `configs/` | 模型架构定义 | yolov8s-p2.yaml |
| `data/` | 数据存储与配置 | visdrone.yaml, images/, labels/ |
| `src/` | 所有可执行代码 | train*.py, evaluate*.py, optimize.py |
| `results/` | 实验结果存储 | *_metrics.json |
| `runs/` | 训练中间产物 | weights/best.pt, results.png |
| `docs/` | 文档知识沉淀 | *.md |
| `ppt/` | 汇报材料 | presentation.md |
| `weights/` | 预训练权重 | yolov8n.pt, yolov8s.pt |

---

# 三、数据层：从原始数据到训练数据

## 3.1 数据流转过程

```
VisDrone 原始数据 (.zip)
    ↓ download_data.py / fast_download.py
解压到 data/visdrone/VisDrone2019-DET-{split}/
    ↓ prepare_data.py
data/visdrone/
├── images/train/    (6,471 张 .jpg)
├── images/val/      (548 张 .jpg)
├── labels/train/    (6,471 个 .txt, YOLO 格式)
├── labels/val/      (548 个 .txt, YOLO 格式)
└── visdrone.yaml    (数据集配置)
```

## 3.2 `visdrone.yaml` — 数据集配置文件

```yaml
path: data/visdrone          # 数据集根目录
train: images/train           # 训练集图片路径 (相对于 path)
val: images/val               # 验证集图片路径
test: images/test             # 测试集图片路径
nc: 10                        # 类别数量
names:                        # 类别名称 (ID 0-9)
  0: pedestrian
  1: people
  2: bicycle
  ...
```

**YOLO 知识点**: Ultralytics 框架通过 YAML 配置文件定位数据集。`path` 是根目录，`train`/`val` 是相对于 `path` 的图片目录路径。标签文件必须与图片文件同名（扩展名为 `.txt`），放在对应的 `labels/` 目录下。

## 3.3 `prepare_data.py` — 数据格式转换

**功能**: 将 VisDrone 原始标注格式转换为 YOLO 格式

**VisDrone 原始格式** (每行一个目标):
```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

**YOLO 格式** (每行一个目标，归一化坐标):
```
<class_id> <x_center> <y_center> <width> <height>
```

**转换公式**:
```python
x_center = (bbox_left + bbox_width / 2) / image_width
y_center = (bbox_top + bbox_height / 2) / image_height
width = bbox_width / image_width
height = bbox_height / image_height
class_id = object_category - 1  # VisDrone 1-10 → YOLO 0-9
```

**YOLO 知识点**: YOLO 格式的坐标是**归一化的中心点坐标+宽高** (xywh)，取值范围 [0, 1]。这与 VisDrone 的**左上角坐标+宽高** (xywh) 不同，需要转换。归一化的好处是与图片分辨率无关，方便不同尺寸图片的统一处理。

## 3.4 `download_data.py` / `fast_download.py` — 数据下载

**功能**: 从网络下载 VisDrone 数据集

**`fast_download.py` 的优化**:
- 使用多线程并发下载
- 支持断点续传
- 自动校验文件完整性

## 3.5 `data/visdrone/labels/` — 标注文件

每个 `.txt` 文件对应一张图片，格式为：
```
0 0.456789 0.123456 0.023456 0.045678
3 0.789012 0.345678 0.056789 0.034567
...
```

**YOLO 知识点**: 每行包含 5 个值：`class_id x_center y_center width height`。多个目标在同一行文件中，每行一个。class_id 从 0 开始编号。

---

# 四、模型层：YOLOv8 架构定义与扩展

## 4.1 Ultralytics 的模型加载机制

YOLOv8 的模型可以通过两种方式加载：

### 方式一: 加载预训练权重
```python
model = YOLO("yolov8s.pt")  # 从 .pt 文件加载
```

### 方式二: 从 YAML 配置创建
```python
model = YOLO("configs/yolov8s-p2.yaml")  # 从 .yaml 文件创建
```

**YOLO 知识点**: YAML 文件定义了模型的**层结构**，每一层指定输入来源、模块类型和参数。Ultralytics 的 `parse_model` 函数会解析 YAML，逐层构建 PyTorch 模块。

## 4.2 `configs/yolov8s-p2.yaml` — 自定义模型配置

这是项目的核心创新文件之一，定义了带 P2 检测头和 CBAM 注意力的 YOLOv8s 变体。

### 4.2.1 配置文件结构

```yaml
nc: 10                              # 类别数
scales:                             # 模型缩放参数
  s: [0.50, 0.50, 1024]            # depth, width, max_channels

backbone:                           # Backbone 定义
  - [from, number, module, args]   # 每层定义

head:                               # Head (Neck + Detect) 定义
  - [from, number, module, args]
```

### 4.2.2 Backbone 逐层解析

```yaml
backbone:
  # ── Stage 1: P1 (1/2 分辨率) ──
  - [-1, 1, Conv, [64, 3, 2]]      # Layer 0: 3→64, 3×3 conv, stride=2
                                     # 输入: 800×800×3 → 输出: 400×400×64

  # ── Stage 2: P2 (1/4 分辨率) ──
  - [-1, 1, Conv, [128, 3, 2]]     # Layer 1: 64→128, stride=2
                                     # 400×400×64 → 200×200×128
  - [-1, 3, C2f, [128, True]]      # Layer 2: C2f 模块, 3 个 bottleneck
                                     # 200×200×128 → 200×200×128
  - [-1, 1, CBAM, []]              # Layer 3: ★ CBAM 注意力 (新增)
                                     # 200×200×128 → 200×200×128
                                     # 通道注意力 + 空间注意力

  # ── Stage 3: P3 (1/8 分辨率) ──
  - [-1, 1, Conv, [256, 3, 2]]     # Layer 4: 128→256, stride=2
                                     # 200×200×128 → 100×100×256
  - [-1, 6, C2f, [256, True]]      # Layer 5: C2f, 6 个 bottleneck
  - [-1, 1, CBAM, []]              # Layer 6: ★ CBAM 注意力 (新增)

  # ── Stage 4: P4 (1/16 分辨率) ──
  - [-1, 1, Conv, [512, 3, 2]]     # Layer 7: 256→512, stride=2
                                     # 100×100×256 → 50×50×512
  - [-1, 6, C2f, [512, True]]      # Layer 8: C2f, 6 个 bottleneck
  - [-1, 1, CBAM, []]              # Layer 9: ★ CBAM 注意力 (新增)

  # ── Stage 5: P5 (1/32 分辨率) ──
  - [-1, 1, Conv, [1024, 3, 2]]    # Layer 10: 512→1024, stride=2
                                     # 50×50×512 → 25×25×1024
  - [-1, 3, C2f, [1024, True]]     # Layer 11: C2f, 3 个 bottleneck
  - [-1, 1, SPPF, [1024, 5]]       # Layer 12: 空间金字塔池化
                                     # 多尺度特征融合
```

**YOLO 知识点**:
- **Conv**: 标准卷积层，`[out_channels, kernel_size, stride]`
- **C2f**: Cross Stage Partial with 2 convolutions，YOLOv8 的核心模块
- **CBAM**: 通道+空间注意力，本项目新增的自定义模块
- **SPPF**: Spatial Pyramid Pooling Fast，多尺度池化融合
- **from=-1**: 表示输入来自上一层
- **scales s**: `[depth=0.50, width=0.50, max_channels=1024]`，控制模型宽度和深度

### 4.2.3 Head 逐层解析

```yaml
head:
  # ── FPN 自顶向下 (高层语义→低层) ──
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # Layer 13: P5 上采样 2×
  - [[-1, 8], 1, Concat, [1]]                     # Layer 14: 拼接 backbone P4 (Layer 8)
  - [-1, 3, C2f, [512]]                           # Layer 15: 融合特征

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # Layer 16: 上采样 2×
  - [[-1, 5], 1, Concat, [1]]                     # Layer 17: 拼接 backbone P3 (Layer 5)
  - [-1, 3, C2f, [256]]                           # Layer 18: P3/8-small

  # ── ★ P2 分支 (新增) ──
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # Layer 19: P3 上采样 2×
  - [[-1, 2], 1, Concat, [1]]                     # Layer 20: 拼接 backbone P2 (Layer 2)
  - [-1, 3, C2f, [128]]                           # Layer 21: P2/4-tiny ★

  # ── PAN 自底向上 (低层位置→高层) ──
  - [-1, 1, Conv, [128, 3, 2]]                    # Layer 22: P2 下采样
  - [[-1, 18], 1, Concat, [1]]                    # Layer 23: 拼接 P3
  - [-1, 3, C2f, [256]]                           # Layer 24: P3 输出

  - [-1, 1, Conv, [256, 3, 2]]                    # Layer 25: P3 下采样
  - [[-1, 15], 1, Concat, [1]]                    # Layer 26: 拼接 head P4
  - [-1, 3, C2f, [512]]                           # Layer 27: P4 输出

  - [-1, 1, Conv, [512, 3, 2]]                    # Layer 28: P4 下采样
  - [[-1, 12], 1, Concat, [1]]                    # Layer 29: 拼接 backbone P5
  - [-1, 3, C2f, [1024]]                          # Layer 30: P5 输出

  # ── 检测头 ──
  - [[21, 24, 27, 30], 1, Detect, [nc]]           # Layer 31: ★ 四层检测头
```

**YOLO 知识点**:
- **FPN (Feature Pyramid Network)**: 自顶向下传递高层语义信息
- **PAN (Path Aggregation Network)**: 自底向上传递低层位置信息
- **Concat**: 沿通道维度拼接特征图
- **Detect**: 检测头，输入多个尺度的特征图，输出边界框和类别预测
- **P2 (160×160)**: 本项目新增的检测层，专门检测 <16px 的小目标

### 4.2.4 标准 vs P2+CBAM 架构对比

| 组件 | 标准 YOLOv8s | P2+CBAM 版本 | 变化 |
|------|-------------|-------------|------|
| Backbone 层数 | 10 | 13 | +3 (CBAM×3) |
| Neck 层数 | 9 | 12 | +3 (P2 分支) |
| 检测层 | P3/P4/P5 | P2/P3/P4/P5 | +1 (P2) |
| 检测点数 | 8,400 | 34,000 | +305% |
| 参数量 | 11.2M | 14.5M | +29% |
| GFLOPs | 28.6 | 47.6 | +66% |

## 4.3 `src/register_custom_modules.py` — CBAM 模块注册

### 4.3.1 为什么需要这个文件？

Ultralytics 解析 YAML 配置时，使用 `globals()[module_name]` 查找模块类。标准模块（Conv, C2f, SPPF 等）已内置在 `ultralytics.nn.tasks` 中。但 CBAM 是自定义模块，需要手动注册。

### 4.3.2 注册机制

```python
def _register():
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    # 将 CBAM 注入到 ultralytics 的全局命名空间
    tasks.CBAM = CBAM
    modules.CBAM = CBAM
```

**YOLO 知识点**: Ultralytics 的 `parse_model` 函数在解析 YAML 时，对每一层执行 `eval(f'{m}(*args)')`，其中 `m` 是模块名。通过将 CBAM 注入到 `ultralytics.nn.tasks` 的全局命名空间，YAML 中的 `CBAM` 就能被正确解析。

### 4.3.3 CBAM 的懒初始化

```python
class CBAM(nn.Module):
    def __init__(self, c1=0, reduction=16, kernel_size=7):
        super().__init__()
        self._built = False  # 标记是否已初始化

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1])  # 根据输入通道数自动初始化
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x
```

**为什么需要懒初始化？**

Ultralytics 在解析 YAML 时，CBAM 不在 `base_modules` 列表中，所以不会自动获得输入通道数 `c1`。YAML 中写 `CBAM, []`（空参数），在第一次前向传播时根据实际输入自动确定通道数。

### 4.3.4 兼容性权重加载

```python
def load_compatible_weights(model, pretrained_path):
    # 加载标准 YOLOv8s 权重
    pretrained = torch.load(pretrained_path, weights_only=False)

    # 逐层匹配形状
    for k, v in pretrained_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            compatible[k] = v  # 形状匹配，加载
        else:
            skipped.append(k)  # 形状不匹配，跳过

    # 只加载匹配的权重，其余随机初始化
    model.load_state_dict(model_state, strict=False)
```

**YOLO 知识点**: 当模型架构发生变化时（如添加 CBAM、增加 P2 检测头），标准预训练权重中没有对应的参数。兼容性加载只加载形状匹配的权重，其余层随机初始化。这比从头训练更快收敛。

## 4.4 `src/cbam.py` — CBAM 独立实现

这是 CBAM 的独立实现文件，包含：

1. **ChannelAttention**: 通道注意力
2. **SpatialAttention**: 空间注意力
3. **CBAM**: 通道+空间注意力组合
4. **CBAMC2f**: 带 CBAM 的 C2f 模块（实验性）

### 4.4.1 通道注意力 (Channel Attention)

```python
class ChannelAttention(nn.Module):
    def forward(self, x):
        # 全局平均池化 + 全局最大池化
        avg_out = self.fc(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.fc(self.max_pool(x))  # [B, C, 1, 1]
        # Sigmoid 激活 → 通道权重
        return self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
```

**YOLO 知识点**: 通道注意力通过全局池化压缩空间信息，学习每个通道的重要性权重。对小目标检测来说，某些通道（如边缘检测通道）可能更重要。

### 4.4.2 空间注意力 (Spatial Attention)

```python
class SpatialAttention(nn.Module):
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        return self.sigmoid(self.conv(combined))         # [B, 1, H, W]
```

**YOLO 知识点**: 空间注意力通过通道维度的池化，学习每个空间位置的重要性权重。对小目标来说，目标所在区域的权重应该更高。

### 4.4.3 `register_custom_modules.py` vs `cbam.py` 的区别

| 文件 | 用途 | 是否被训练脚本使用 |
|------|------|------------------|
| `cbam.py` | CBAM 的独立实现，含文档和 CBAMC2f | 否 (参考实现) |
| `register_custom_modules.py` | CBAM 注册到 ultralytics，含懒初始化和权重加载 | **是** (训练脚本导入) |

---

# 五、训练层：三个训练脚本的设计逻辑

## 5.1 `src/train.py` — 基线训练脚本

### 5.1.1 设计思路

基线训练脚本采用**命令行参数驱动**的设计，支持灵活配置：

```python
parser.add_argument("--model", default="yolov8n")   # 模型选择
parser.add_argument("--epochs", default=100)          # 训练轮数
parser.add_argument("--imgsz", default=640)           # 输入尺寸
parser.add_argument("--batch", default=16)            # 批量大小
```

### 5.1.2 训练参数详解

| 参数 | 值 | YOLO 知识点 |
|------|-----|------------|
| `optimizer="auto"` | 自动选择 | Ultralytics 根据配置自动选择 SGD 或 AdamW |
| `lr0=0.01` | 初始学习率 | SGD 的典型初始学习率 |
| `lrf=0.01` | 最终学习率比例 | 最终学习率 = lr0 × lrf = 0.0001 |
| `momentum=0.937` | SGD 动量 | 加速收敛，抑制振荡 |
| `weight_decay=0.0005` | 权重衰减 | L2 正则化，防止过拟合 |
| `warmup_epochs=3` | 预热轮数 | 前 3 个 epoch 线性增加学习率 |
| `box=7.5` | 边界框损失权重 | L_box 的权重系数 |
| `cls=0.5` | 分类损失权重 | L_cls 的权重系数 |
| `dfl=1.5` | DFL 损失权重 | L_dfl 的权重系数 |
| `close_mosaic=10` | 关闭 Mosaic 的轮数 | 最后 10 个 epoch 关闭 Mosaic 增强 |
| `amp=True` | 混合精度训练 | 使用 FP16 加速，节省显存 |

**YOLO 知识点**: `close_mosaic` 是一个重要的训练技巧。Mosaic 增强在训练前期增加数据多样性，但在训练后期可能导致目标被过度裁剪。最后几个 epoch 关闭 Mosaic，让模型在 "干净" 的数据上收敛。

## 5.2 `src/train_improved.py` — 改进训练脚本

### 5.2.1 与基线的差异

| 参数 | 基线 | 改进 | 变化原因 |
|------|------|------|---------|
| 模型 | yolov8n | yolov8s | 更大容量 |
| imgsz | 640 | 800 | 保留更多小目标信息 |
| batch | 16 | 8 | 更大图片需要更小 batch |
| epochs | 50 | 80 | 更充分训练 |
| warmup_epochs | 3 | 5 | 更长预热 |
| close_mosaic | 10 | 15 | 更晚关闭 |
| hsv_h | 0.015 | 0.02 | 更强色调增强 |
| hsv_s | 0.7 | 0.8 | 更强饱和度增强 |
| hsv_v | 0.4 | 0.5 | 更强亮度增强 |
| degrees | 0 | 10.0 | 新增旋转增强 |
| translate | 0.1 | 0.15 | 更强平移增强 |
| scale | 0.5 | 0.6 | 更强缩放增强 |
| shear | 0 | 5.0 | 新增剪切增强 |
| perspective | 0 | 0.001 | 新增透视增强 |
| mixup | 0 | 0.15 | 新增 MixUp |
| copy_paste | 0 | 0.1 | 新增 CopyPaste |
| erasing | 0 | 0.5 | 新增随机擦除 |

### 5.2.2 关键代码差异

```python
# 基线: 使用命令行参数
model = YOLO(f"{args.model}.pt")

# 改进: 直接指定更大的模型
model = YOLO("yolov8s.pt")

# 改进: 更强的数据增强
results = model.train(
    mixup=0.15,        # MixUp: 两张图按比例混合
    copy_paste=0.1,    # CopyPaste: 复制粘贴目标
    erasing=0.5,       # 随机擦除: 随机遮挡区域
    degrees=10.0,      # 旋转: ±10度
    shear=5.0,         # 剪切: ±5度
    perspective=0.001, # 透视变换
)
```

**YOLO 知识点**: MixUp 和 CopyPaste 是两种强大的数据增强方法：
- **MixUp**: 将两张图按比例混合 `new_img = α × img1 + (1-α) × img2`，增加样本多样性
- **CopyPaste**: 将一张图中的目标复制粘贴到另一张图，直接增加目标数量

## 5.3 `src/train_p2.py` — P2+CBAM 训练脚本

### 5.3.1 与前两个脚本的关键区别

```python
# 1. 导入自定义模块注册 (必须在 YOLO 之前)
import register_custom_modules

# 2. 修复 pi_heif 问题
_pil_open = PIL.Image.open  # 保存原始方法
# ... import ultralytics ...
PIL.Image.open = _pil_open  # 恢复

# 3. 从 YAML 创建自定义模型
model = YOLO(model_yaml)  # 不是 .pt 文件

# 4. 加载兼容性权重
register_custom_modules.load_compatible_weights(model.model, str(weights_path))

# 5. 从头训练 (pretrained=False，因为已手动加载)
results = model.train(pretrained=False, ...)
```

### 5.3.2 训练流程对比

```
基线/改进训练:
  YOLO("yolov8n.pt") → model.train() → 自动加载 COCO 权重 → 训练

P2+CBAM 训练:
  import register_custom_modules        # 注册 CBAM
  YOLO("configs/yolov8s-p2.yaml")       # 从 YAML 创建模型
  load_compatible_weights(model, ...)    # 手动加载兼容权重
  model.train(pretrained=False)          # 训练 (不自动加载)
```

### 5.3.3 pi_heif 问题修复

```python
import PIL.Image
_pil_open = PIL.Image.open  # 保存原始 PIL.Image.open

# 导入 ultralytics (会 monkey-patch PIL.Image.open)
from ultralytics import YOLO

PIL.Image.open = _pil_open  # 恢复原始方法
```

**问题原因**: Ultralytics 在导入时尝试导入 `pi_heif` (HEIF 图片格式支持)，这会修改 `PIL.Image.open`。但 Windows 上 pi_heif 安装困难，导致修改失败。

---

# 六、评估层：多维度评估体系

## 6.1 `src/evaluate.py` — 单模型评估

### 6.1.1 功能

对单个模型在验证集上评估，输出 mAP、Precision、Recall 和各类别 AP。

### 6.1.2 评估流程

```python
model = YOLO(weights_path)           # 加载模型
metrics = model.val(                  # 在 val 集上评估
    data="data/visdrone/visdrone.yaml",
    imgsz=640,
    conf=0.001,                       # 极低阈值，评估所有预测
    iou=0.6,                          # NMS IoU 阈值
)

# 提取指标
mAP50 = metrics.box.map50            # mAP@0.5
mAP50_95 = metrics.box.map           # mAP@0.5:0.95
precision = metrics.box.mp           # Precision
recall = metrics.box.mr              # Recall
per_class_ap = metrics.box.ap50      # 各类别 AP@0.5
```

**YOLO 知识点**: `conf=0.001` 是评估时的置信度阈值。使用极低阈值是为了评估模型的**完整性能**，不受阈值选择的影响。实际部署时再根据需求调整阈值。

### 6.1.3 输出文件

```json
{
  "mAP50": 0.2979,
  "mAP50_95": 0.1663,
  "precision": 0.4237,
  "recall": 0.3191,
  "per_class_ap50": {
    "pedestrian": 0.312,
    "people": 0.243,
    "bicycle": 0.062,
    ...
  }
}
```

## 6.2 `src/evaluate_all.py` — 标准化全量评估

### 6.2.1 设计思路

统一评估所有模型（基线、改进、P2+CBAM），生成标准化对比报告。

### 6.2.2 模型配置

```python
MODELS = {
    "baseline": {
        "name": "YOLOv8n Baseline",
        "weights": "runs/baseline/.../best.pt",
        "imgsz": 640,
    },
    "improved": {
        "name": "YOLOv8s Improved",
        "weights": "runs/improved/.../best.pt",
        "imgsz": 800,
    },
    "p2_cbam": {
        "name": "YOLOv8s P2+CBAM",
        "weights": "runs/p2/.../best.pt",
        "imgsz": 800,
    },
}
```

### 6.2.3 输出

- `results/standardized_eval.json`: 机器可读的评估结果
- `results/standardized_eval.md`: 人可读的对比报告

## 6.3 `src/sahi_eval.py` — SAHI 评估

### 6.3.1 SAHI 评估流程

```python
# 1. 构建 COCO ground truth
coco_gt_dict = build_coco_gt(val_image_dir, val_label_dir)

# 2. 加载 SAHI 模型
model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=str(weights_path),
    confidence_threshold=0.25,
    device="cuda:0",
)

# 3. 对每张图片进行切片推理
for img_file in val_images:
    result = get_sliced_prediction(
        image=str(img_file),
        detection_model=model,
        slice_height=640,            # 切片高度
        slice_width=640,             # 切片宽度
        overlap_height_ratio=0.2,    # 重叠率
        overlap_width_ratio=0.2,
        postprocess_type="NMS",      # 后处理: NMS
        postprocess_match_threshold=0.5,
    )

# 4. COCOeval 评估
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
```

**YOLO 知识点**: SAHI (Slicing Aided Hyper Inference) 的核心是 `get_sliced_prediction` 函数，它将大图切成重叠小块，分别推理后合并结果。`overlap_height_ratio=0.2` 表示相邻切片有 20% 的重叠区域。

---

# 七、优化层：零训练优化流水线

## 7.1 `src/optimize.py` — 零训练优化脚本

这是项目中最复杂的脚本，包含三个独立的优化实验。

### 7.1.1 三大实验

```
optimize.py
├── run_sahi_tta()          # 实验 1: SAHI + 测试时增强
├── run_threshold_search()  # 实验 2: 置信度阈值网格搜索
├── run_ensemble()          # 实验 3: 多模型集成
└── main()                  # 主函数: 运行所有实验并汇总
```

### 7.1.2 COCO Ground Truth 构建

```python
def build_coco_gt(val_image_dir, val_label_dir):
    """从 YOLO 标注构建 COCO 格式的 ground truth"""
    coco = {"images": [], "annotations": [], "categories": []}

    for img_id, img_file in enumerate(img_files):
        # 图片信息
        coco["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": w, "height": h
        })

        # 标注信息 (YOLO 格式 → COCO 格式)
        for line in label_file:
            cls_id, x_c, y_c, bw, bh = parse_yolo_label(line)
            x = (x_c - bw / 2) * w    # 中心→左上角
            y = (y_c - bh / 2) * h
            coco["annotations"].append({
                "bbox": [x, y, bw * w, bh * h],  # COCO xywh
                "area": bw * w * bh * h,
                "category_id": cls_id,
            })
```

**YOLO 知识点**: COCO 格式的 bbox 是 `[x_top_left, y_top_left, width, height]`，而 YOLO 格式是 `[x_center, y_center, width, height]`（归一化）。需要转换后才能使用 pycocotools 评估。

### 7.1.3 实验 1: SAHI + TTA

```python
def sahi_tta_predict(sahi_model, image_path):
    # 原图推理
    result1 = get_sliced_prediction(image=image_path, ...)
    preds1 = [pred_to_coco(p) for p in result1.object_prediction_list]

    # 水平翻转推理
    img_flipped = cv2.flip(img, 1)
    result2 = get_sliced_prediction(image=flipped_path, ...)

    # 翻转坐标还原
    for pred in result2.object_prediction_list:
        bbox = pred.bbox.to_coco_bbox()
        x_orig = w - bbox[0] - bbox[2]  # 水平翻转还原
        preds2.append(...)

    # NMS 合并
    return nms_per_class(preds1 + preds2, iou_thresh=0.5)
```

**YOLO 知识点**: TTA (Test Time Augmentation) 通过对输入进行增强（如水平翻转），将增强后的结果与原图结果合并。翻转后的坐标需要还原：`x_orig = width - x_flip - bbox_width`。

### 7.1.4 实验 2: 置信度阈值网格搜索

```python
def run_threshold_search(sahi_model, ...):
    # 1. 收集所有预测 (conf=0.05，保留尽可能多的预测)
    all_raw_preds = collect_predictions(sahi_model, conf=0.05)

    # 2. 按不同阈值过滤并评估
    for conf in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        filtered = [p for p in all_raw_preds if p["score"] >= conf]
        mAP = evaluate_coco(coco_gt, filtered)
        results_grid.append({"conf": conf, "mAP50": mAP})

    # 3. 找到最优阈值
    best = max(results_grid, key=lambda x: x["mAP50"])
```

**YOLO 知识点**: 置信度阈值控制检测框的过滤。降低阈值会增加 Recall（检测到更多目标）但可能降低 Precision（引入更多误检）。mAP 是 PR 曲线下面积，最优阈值是使 PR 曲线面积最大的阈值。

### 7.1.5 实验 3: 多模型集成

```python
def run_ensemble(model1_path, model2_path, ...):
    model1 = YOLO(model1_path)  # YOLOv8n (权重 0.7)
    model2 = YOLO(model2_path)  # YOLOv8s (权重 1.0)

    for img_file in img_files:
        # 两个模型分别推理
        r1 = model1.predict(source=img_str, conf=0.15)
        r2 = model2.predict(source=img_str, conf=0.15)

        # 加权融合
        for box in r1.boxes:
            score = float(box.conf[0]) * 0.7  # 基线模型权重低
            coco_results.append(...)
        for box in r2.boxes:
            score = float(box.conf[0]) * 1.0  # 改进模型权重高
            coco_results.append(...)

    # NMS 去重
    coco_results = nms_per_class(coco_results)
```

---

# 八、可视化层：检测结果的可视化展示

## 8.1 `src/visualize.py` — 可视化脚本

### 8.1.1 三个功能

```bash
# 检测结果可视化
python src/visualize.py detect --weights best.pt --source data/visdrone/images/val

# GradCAM 热力图
python src/visualize.py heatmap --weights best.pt --source img.jpg

# 多模型对比
python src/visualize.py compare --models baseline=best1.pt improved=best2.pt --source val/
```

### 8.1.2 检测对比图生成

```python
def compare_models(model_paths, img_dir, output_dir):
    models = {name: YOLO(path) for name, path in model_paths.items()}

    for img_file in img_files:
        images = []
        for name, model in models.items():
            result = model.predict(source=img_file, conf=0.25)
            annotated = result.plot(conf=True, labels=True, boxes=True)
            cv2.putText(annotated, name, ...)  # 添加模型名称
            images.append(annotated)

        # 水平拼接所有模型的结果
        combined = np.hstack(resized_images)
        cv2.imwrite(save_path, combined)
```

## 8.2 `src/utils.py` — 工具函数

### 8.2.1 核心功能

| 函数 | 功能 | 使用场景 |
|------|------|---------|
| `yolo_to_xyxy()` | YOLO 格式 → 像素坐标 | 坐标转换 |
| `draw_boxes()` | 在图片上绘制标注框 | 数据可视化 |
| `visualize_samples()` | 随机可视化数据集样本 | 数据集检查 |
| `count_labels()` | 统计各类别目标数量 | 数据集分析 |

### 8.2.2 类别颜色映射

```python
CLASS_COLORS = [
    (255, 0, 0),     # pedestrian - 蓝
    (0, 255, 0),     # people - 绿
    (0, 0, 255),     # bicycle - 红
    (255, 255, 0),   # car - 青
    ...
]
```

---

# 九、文档层：完整的知识沉淀体系

## 9.1 文档分类

| 类型 | 文档 | 目标读者 |
|------|------|---------|
| **使用指南** | usage_guide.md | 想复现项目的人 |
| **技术详解** | yolo_deep_dive.md | 想深入理解 YOLO 的人 |
| **实验记录** | experiments/exp*.md | 需要实验细节的人 |
| **总结报告** | experiment_report.md, technical_report.md | 写报告/论文的人 |
| **学习反思** | learning_summary.md | 想了解项目思路的人 |
| **论文调研** | paper_review.md | 做文献调研的人 |
| **架构文档** | project_architecture.md (本文) | 想理解项目结构的人 |
| **汇报材料** | presentation.md | 答辩/汇报用 |

## 9.2 文档间的引用关系

```
README.md
├── 使用教程 (usage_guide.md)
│   └── 引用: 所有 src/ 脚本的使用方法
├── YOLO 详解 (yolo_deep_dive.md)
│   └── 引用: YOLO 架构、损失函数、训练流程
├── 学习总结 (learning_summary.md)
│   └── 引用: 五轮实验的设计思路和反思
├── 分实验报告 (experiments/exp*.md)
│   └── 引用: 每轮实验的详细配置和结果
├── 技术报告 (technical_report.md)
│   └── 引用: 所有实验结果的汇总
└── 架构文档 (project_architecture.md)
    └── 引用: 所有文件的作用和关系
```

---

# 十、数据流全景图

## 10.1 训练阶段数据流

```
VisDrone .zip
    ↓ download_data.py
原始图片 + 原始标注
    ↓ prepare_data.py
YOLO 格式标注 + visdrone.yaml
    ↓ train.py / train_improved.py / train_p2.py
    │   ↓ YOLO(model.pt / model.yaml)
    │   ↓ model.train(data=visdrone.yaml)
    │   ↓ 每 epoch:
    │   │   ├── 数据加载 (images/ + labels/)
    │   │   ├── 数据增强 (Mosaic, MixUp, ...)
    │   │   ├── 前向传播 (Backbone → Neck → Head)
    │   │   ├── 损失计算 (L_box + L_cls + L_dfl)
    │   │   └── 反向传播 (梯度更新)
    │   └── 每 epoch 结束:
    │       ├── 在 val 集上评估
    │       ├── 保存 best.pt (如果 mAP 提升)
    │       └── 记录 results.csv
    ↓
runs/xxx/weights/best.pt          (模型权重)
runs/xxx/results.png              (训练曲线)
runs/xxx/confusion_matrix.png     (混淆矩阵)
```

## 10.2 评估阶段数据流

```
runs/xxx/weights/best.pt          (训练好的模型)
    ↓ evaluate.py / evaluate_all.py
    │   ↓ model.val(data=visdrone.yaml)
    │   ↓ Ultralytics 内置评估:
    │   │   ├── 加载 val 集图片
    │   │   ├── 前向传播 → 预测
    │   │   ├── NMS 后处理
    │   │   ├── 与 GT 匹配 (IoU)
    │   │   └── 计算 mAP
    │   ↓ 提取指标
    ↓
results/xxx_metrics.json          (评估指标 JSON)
results/standardized_eval.md      (对比报告)
```

## 10.3 SAHI 优化阶段数据流

```
runs/improved/.../best.pt         (改进模型)
    ↓ sahi_eval.py / optimize.py
    │   ↓ 对每张 val 图片:
    │   │   ├── 切片: 2000×1500 → 多个 640×640
    │   │   ├── 每个切片独立推理
    │   │   └── NMS 合并跨切片结果
    │   ↓ 收集所有图片的预测
    │   ↓ pycocotools COCOeval 评估
    ↓
results/sahi_metrics.json         (SAHI 评估结果)
results/threshold_search.json     (阈值搜索结果)
results/sahi_tta_metrics.json     (TTA 评估结果)
results/optimization_summary.json (综合对比)
```

---

# 十一、文件依赖关系矩阵

## 11.1 脚本依赖关系

| 脚本 | 依赖的配置 | 依赖的数据 | 依赖的权重 | 依赖的其他脚本 |
|------|-----------|-----------|-----------|--------------|
| `prepare_data.py` | - | data/zips/ | - | - |
| `train.py` | visdrone.yaml | data/visdrone/ | yolov8n.pt | - |
| `train_improved.py` | visdrone.yaml | data/visdrone/ | yolov8s.pt | - |
| `train_p2.py` | visdrone.yaml, yolov8s-p2.yaml | data/visdrone/ | yolov8s.pt | register_custom_modules.py |
| `evaluate.py` | visdrone.yaml | data/visdrone/ | best.pt | - |
| `evaluate_all.py` | visdrone.yaml | data/visdrone/ | 所有 best.pt | register_custom_modules.py |
| `sahi_eval.py` | visdrone.yaml | data/visdrone/ | best.pt | - |
| `optimize.py` | - | data/visdrone/ | baseline + improved best.pt | - |
| `detect.py` | - | 任意图片/视频 | best.pt | - |
| `visualize.py` | - | data/visdrone/ | best.pt | - |
| `utils.py` | - | data/visdrone/ | - | - |

## 11.2 数据流向图

```
prepare_data.py ──→ data/visdrone/ ──→ train.py ──→ runs/ ──→ evaluate.py ──→ results/
                                  ──→ train_improved.py ──→ runs/ ──→ sahi_eval.py ──→ results/
                                  ──→ train_p2.py ──→ runs/ ──→ evaluate_all.py ──→ results/
                                                     runs/ ──→ optimize.py ──→ results/
                                  ──→ visualize.py ──→ results/
```

---

# 十二、YOLO 知识在项目中的应用映射

## 12.1 YOLO 核心概念 → 项目文件

| YOLO 概念 | 项目中的体现 | 相关文件 |
|----------|------------|---------|
| Backbone (特征提取) | CSPDarknet + C2f | configs/yolov8s-p2.yaml (Layer 0-12) |
| Neck (特征融合) | PANet + FPN | configs/yolov8s-p2.yaml (Layer 13-30) |
| Head (检测输出) | Decoupled Anchor-Free | configs/yolov8s-p2.yaml (Layer 31) |
| 损失函数 | CIoU + BCE + DFL | train.py (box=7.5, cls=0.5, dfl=1.5) |
| 数据增强 | Mosaic, MixUp, CopyPaste | train_improved.py |
| NMS | 非极大值抑制 | optimize.py (nms_per_class) |
| mAP | 平均精度均值 | evaluate.py, sahi_eval.py |
| Anchor-Free | 无锚框设计 | YOLOv8 内置 |
| Decoupled Head | 解耦检测头 | YOLOv8 内置 |

## 12.2 小目标检测技术 → 项目实现

| 技术 | 原理 | 项目实现 |
|------|------|---------|
| 更大输入 | 保留更多小目标像素 | train_improved.py (imgsz=800) |
| P2 检测头 | 160×160 特征图检测 <16px 目标 | configs/yolov8s-p2.yaml |
| CBAM 注意力 | 通道+空间注意力聚焦小目标 | register_custom_modules.py |
| SAHI 切片 | 切片推理保持原始分辨率 | sahi_eval.py, optimize.py |
| CopyPaste | 增加小目标训练样本 | train_improved.py (copy_paste=0.1) |
| 低置信度阈值 | 保留更多低置信度检测 | optimize.py (conf=0.05) |
| TTA | 测试时增强增加召回 | optimize.py (sahi_tta_predict) |

## 12.3 模型评估知识 → 项目实现

| 评估概念 | 项目实现 | 说明 |
|---------|---------|------|
| IoU | pycocotools COCOeval | 交并比计算 |
| Precision-Recall | COCOeval.evaluate() | PR 曲线 |
| AP | COCOeval.accumulate() | 单类别平均精度 |
| mAP | COCOeval.stats[1] | 所有类别 AP 均值 |
| mAP@0.5:0.95 | COCOeval.stats[0] | 多 IoU 阈值平均 |
| Per-class AP | eval["precision"] 切片 | 各类别单独评估 |

---

## 结语

本项目通过 15 个 Python 脚本、1 个 YAML 配置、7 个 JSON 结果文件和 12 个 Markdown 文档，构建了一个完整的 YOLOv8 小目标检测实验系统。

每个文件都有明确的职责，文件之间通过标准接口（文件路径、JSON 格式、YAML 配置）连接，形成了清晰的数据流和控制流。

理解这个架构，不仅有助于复现实验，更能帮助理解目标检测系统从数据到模型到评估的完整链路。
