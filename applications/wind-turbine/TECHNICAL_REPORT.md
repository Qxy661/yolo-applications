# 风电场叶片缺陷检测技术报告

> 基于 YOLOv11 的风力发电机叶片表面缺陷检测系统
> 版本: v1.0 | 日期: 2026-05-18

---

## 目录

1. [项目概述](#1-项目概述)
2. [文献综述](#2-文献综述)
3. [数据集调研与方案](#3-数据集调研与方案)
4. [技术方案设计](#4-技术方案设计)
5. [模型架构](#5-模型架构)
6. [实验设计](#6-实验设计)
7. [项目结构与规范](#7-项目结构与规范)
8. [参考文献](#8-参考文献)

---

## 1. 项目概述

### 1.1 研究背景

风力发电机叶片长期暴露在恶劣环境中，面临风沙侵蚀、雷击、紫外线老化等威胁。叶片缺陷若不及时检测，可能导致停机甚至安全事故。传统人工巡检效率低、成本高、主观性强，基于深度学习的自动化检测成为主流方向。

### 1.2 研究目标

构建基于 YOLOv11 的叶片缺陷检测系统，支持 **5 类核心缺陷** 识别：

| 类别ID | 中文名称 | 英文名称 | 视觉特征 | 检测难度 |
|--------|---------|---------|---------|---------|
| 0 | 叶片裂纹 | Crack | 细长线条状，宽度<1mm | ★★★★★ |
| 1 | 叶片破损 | Breakage | 不规则缺损区域，孔洞 | ★★★ |
| 2 | 雷击损伤 | Lightning | 烧蚀痕迹，碳化变色 | ★★★★ |
| 3 | 涂层脱落 | Peeling | 表面涂层剥离，露出基材 | ★★★ |
| 4 | 边缘侵蚀 | Erosion | 前缘磨损，厚度减薄 | ★★★★ |

### 1.3 性能目标

| 指标 | 基线目标 | 最终目标 |
|------|---------|---------|
| mAP@0.5 | > 0.70 | > 0.85 |
| mAP@0.5:0.95 | > 0.50 | > 0.65 |
| FPS (NVIDIA GPU) | > 30 | > 50 |
| 模型大小 | < 50MB | < 20MB |

---

## 2. 文献综述

### 2.1 YOLO系列在叶片检测中的应用

| 论文 | 年份 | 基线模型 | 改进方法 | 关键结果 |
|------|------|---------|---------|---------|
| SOD-YOLO (Zhang & Wen) | 2022 | YOLOv5 | CBAM注意力 + 微尺度检测 | mAP 95.1%, +7.82% |
| WTBD-YOLOv8 (Tong et al.) | 2024 | YOLOv8 | GhostCBS + MHSA + Mini-BiFPN | AP 98.3%, 参数↓38.2% |
| LE-YOLO (Fu et al.) | 2024 | YOLOv7 | GSConv + SimAM + EIoU | mAP 78.7%, 105.1 FPS |
| GCB-YOLO (Zhang et al.) | 2025 | YOLOv5s | GhostNet + CA + BiFPN | mAP@0.5 94.72%, 7.5MB |
| Davis et al. | 2024 | YOLOv9-C | ResNet18-FPN | mAP50 0.849 |
| RSD-YOLO (Wang et al.) | 2025 | 改进YOLO | 多尺度特征融合 | FWCI 20.44 (top 1%) |

### 2.2 注意力机制对比

| 注意力 | 类型 | 参数开销 | 位置感知 | 小缺陷效果 | 适用场景 |
|--------|------|---------|---------|-----------|---------|
| SE | 通道 | 极低 (0.1M) | 无 | 一般 | 通用检测 |
| CBAM | 通道+空间 | 低 (0.2M) | 弱 | **可能有害** | 大中目标 |
| CA | 通道+位置 | 低 (0.2M) | 有 | **推荐** | 小缺陷检测 |
| ECA | 通道(轻量) | 极低 (<0.1M) | 无 | 轻量推荐 | 边缘部署 |
| SimAM | 无参数 | 无 | 有 | 推荐 | 轻量场景 |
| C2PSA | 位置敏感注意力 | 中 (0.5M) | 有 | 推荐 | YOLOv11+ |

**关键发现**: CBAM的空间注意力可能抑制小缺陷特征（同VisDrone小目标实验结论），建议优先使用CA或ECA。

### 2.3 技术趋势总结

1. **轻量化**: GhostNet/GSConv替代标准卷积，参数量↓30-50%
2. **多尺度融合**: BiFPN/PAFPN替代标准FPN，小缺陷检测提升显著
3. **注意力机制**: CA/SimAM优于CBAM（小缺陷场景）
4. **数据增强**: Mosaic+MixUp+CopyPaste组合效果最佳
5. **切片推理**: SAHI对小缺陷检测提升10-15%（同低空小目标结论）
6. **端到端检测**: NMS-Free (YOLOv10) 减少后处理延迟

---

## 3. 数据集调研与方案

### 3.1 已发现的公开数据集

经过全面搜索（GitHub/HuggingFace/Kaggle/Roboflow），发现以下可获取的数据集：

#### 数据集A: 风电叶片表面缺陷数据集 (StyleGAN合成)

| 属性 | 详情 |
|------|------|
| 来源 | GitHub: zhaowenhai2023/Wind-turbine-blade-surface-defect-dataset |
| 图片数 | 3,800+ 张高分辨率图像 |
| 生成方式 | StyleGAN3 + PBGM 合成 |
| 标注格式 | YOLO txt |
| 下载方式 | **百度网盘** (需手动下载) |
| 图片链接 | https://pan.baidu.com/s/1_tJBlhuNE1eDMxTO9uzdWA (提取码: 1234) |
| 标注链接 | https://pan.baidu.com/s/1Qr0jRVLFUpa13HH0HDsC5Q (提取码: 1234) |
| 优点 | 数量充足，格式规范 |
| 缺点 | 合成数据，非真实巡检图像；类别未明确列出 |
| Star | 26 |

#### 数据集B: Blade30 无人机巡检数据集

| 属性 | 详情 |
|------|------|
| 来源 | GitHub: cong-yang/Blade30 |
| 论文 | Renewable Energy (2023) |
| 图片数 | 1,302 张真实无人机巡检图像 (30个叶片) |
| 标注格式 | JSON + PNG分割掩码 |
| 缺陷类别 | Defects, Contaminations |
| 下载方式 | **Google Drive / OneDrive** |
| Google Drive (叶片1-15) | https://drive.google.com/file/d/1HbB4t9xV2oCgSSxR9hMEOU6v9qDfetmR/view |
| Google Drive (叶片16-30) | https://drive.google.com/file/d/1SwRdMzA7zCkNVlHuWvk8uK6eDToM0mUV/view |
| OneDrive (完整) | https://1drv.ms/u/s!AoXJBmXKVWu5tmtUzCJULhrtYuIP?e=KYOtlo |
| 优点 | 真实无人机图像，学术论文支撑 |
| 缺点 | 分割格式(需转检测)，仅2类缺陷 |

#### 数据集C: 风电叶片分类数据集 (6类)

| 属性 | 详情 |
|------|------|
| 来源 | GitHub: princebhanusteta/Wind-turbine-defect-classification |
| 图片数 | ~1,000+ 张 (裁剪后的缺陷区域) |
| 缺陷类别 | crack, corrosion, surface_injury, thunderstrike, hidden_crack 等6类 |
| 标注格式 | VOC XML + CSV |
| 下载方式 | **仓库内 dataRaw/ 目录 (需Git LFS)** |
| 优点 | 包含thunderstrike(雷击)类别，与我们需求匹配度高 |
| 缺点 | 分类数据集(需转检测)，需裁剪处理 |

#### 数据集D: fans-defect-Dataset (3类)

| 属性 | 详情 |
|------|------|
| 来源 | GitHub: mxy021120-ops/fans-defect-Dataset |
| 图片数 | 4,802 张 |
| 缺陷类别 | Dirt, Oil Leakage, Pin Hole |
| 标注格式 | YOLO txt + data.yaml |
| 下载方式 | **Git Clone** |
| 优点 | YOLO格式直接可用，数量充足 |
| 缺点 | 类别与我们目标不匹配(风扇缺陷，非叶片) |

### 3.2 与目标5类的匹配分析

| 目标类别 | 数据集A | 数据集B | 数据集C | 数据集D |
|---------|---------|---------|---------|---------|
| Crack (裂纹) | 待确认 | - | 有 | - |
| Breakage (破损) | 待确认 | 有(Defects) | 有 | - |
| Lightning (雷击) | 待确认 | - | 有(thunderstrike) | - |
| Peeling (脱漆) | 待确认 | - | 有 | - |
| Erosion (侵蚀) | 待确认 | 有(Contaminations) | 有(corrosion) | - |

**结论**: 无单个数据集覆盖全部5类。最佳策略是 **多源数据合并**。

### 3.3 推荐数据集获取方案

**优先级1: 数据集A (百度网盘)**
- 3,800+张合成图像，数量最充足
- 需手动从百度网盘下载
- 下载后需确认具体类别

**优先级2: 数据集B (Google Drive)**
- 1,302张真实无人机图像
- 需从Google Drive下载
- 需要JSON→YOLO格式转换

**优先级3: 数据集C (Git仓库)**
- 包含thunderstrike(雷击)类别
- 需从GitHub仓库获取
- 需要分类→检测格式转换

**优先级4: 自建数据**
- 使用无人机实地采集
- 专业标注工具(LabelImg/CVAT)标注
- 可确保覆盖全部5类

---

## 4. 技术方案设计

### 4.1 整体技术路线

```
文献调研 → 数据集构建 → YOLOv11基线训练 → 改进优化 → 部署评估
   ↓           ↓              ↓               ↓            ↓
 论文精读   多源数据合并    ultralytics      注意力机制    ONNX导出
 5类缺陷   YOLO格式统一    预训练权重        数据增强      推理优化
```

### 4.2 基线模型选择

| 配置项 | 选择 | 理由 |
|--------|------|------|
| 模型 | YOLOv11n / YOLOv11s | ultralytics最新，架构成熟 |
| 预训练 | COCO 预训练 | 迁移学习，收敛更快 |
| 输入尺寸 | 640×640 | 标准尺寸，平衡精度与速度 |
| 优化器 | AdamW | 自适应学习率，收敛稳定 |
| 学习率 | 0.001 (初始) | 配合余弦退火 |
| 批量大小 | 16 (nano) / 8 (small) | 显存适配 NVIDIA GPU |
| 训练轮次 | 100 epochs | 配合早停(patience=30) |

### 4.3 改进方案（迭代优化）

#### 迭代一：数据增强优化
- 策略: Mosaic(1.0) + MixUp(0.15) + CopyPaste(0.3) + 颜色抖动
- 预期提升: +5-10%

#### 迭代二：注意力机制集成
- 方案: 在C2f模块后添加CA注意力
- 位置: Backbone最后一层 + Neck输出层
- 预期提升: +2-3%

#### 迭代三：Neck增强
- 方案: 替换标准FPN为BiFPN/PAFPN
- 预期提升: +3-5%

#### 迭代四：SAHI切片推理
- 参数: 切片尺寸640, overlap=0.2, conf=0.2
- 预期提升: +10-15%

#### 迭代五：综合改进
- YOLOv11s + CA + BiFPN + 数据增强 + SAHI
- 预期最终mAP: > 0.85

---

## 5. 模型架构

### 5.1 YOLOv11 架构概览

```
Input (640×640×3)
    ↓
Backbone:
  Conv→Conv→C3k2→Conv→C3k2→Conv→C3k2→Conv→C3k2→SPPF
  [64]  [128]  [128]  [256]  [256]  [512]  [512]  [1024] [1024]
    ↓
Neck (FPN + PAN):
  Upsample→Concat→C3k2→Upsample→Concat→C3k2
  Conv→Concat→C3k2→Conv→Concat→C3k2
    ↓
Head (Decoupled + Anchor-Free):
  P3 (80×80) → Detect → 小缺陷
  P4 (40×40) → Detect → 中缺陷
  P5 (20×20) → Detect → 大缺陷
    ↓
Output: Bounding Boxes + Classes + Confidence
```

### 5.2 改进点: CA注意力机制

```
Feature Map (H×W×C)
    ↓
┌─────────────────────────┐
│  X-direction AvgPool    │  → (H×1×C)
│  Y-direction AvgPool    │  → (1×W×C)
│  Concat → Conv → Split  │
│  Sigmoid → Attention    │
└─────────────────────────┘
    ↓
Weighted Feature Map (H×W×C)
```

**CA vs CBAM 对比**:
- CBAM: 空间注意力聚焦"显著"区域 → 小缺陷被抑制
- CA: 保留X/Y位置信息 → 小缺陷不被抑制

---

## 6. 实验设计

### 6.1 实验计划

| 实验 | 目标 | 模型 | 改进 | 预期mAP |
|------|------|------|------|---------|
| E1: 基线 | 建立基准 | YOLOv11n | 无 | ~0.70 |
| E2: 数据增强 | 数据层优化 | YOLOv11n | Mosaic+MixUp+CopyPaste | ~0.75 |
| E3: 注意力 | 特征增强 | YOLOv11n | +CA注意力 | ~0.78 |
| E4: Neck增强 | 多尺度融合 | YOLOv11n | +BiFPN | ~0.80 |
| E5: SAHI | 推理优化 | YOLOv11n | +SAHI切片 | ~0.85 |
| E6: 综合 | 全部改进 | YOLOv11s | 所有改进 | >0.85 |

### 6.2 评估指标

| 指标类型 | 指标 | 说明 |
|---------|------|------|
| 主要指标 | mAP@0.5 | IoU=0.5时的平均精度 |
| 主要指标 | mAP@0.5:0.95 | IoU=0.5:0.95的平均精度 |
| 辅助指标 | Precision | 精确率 |
| 辅助指标 | Recall | 召回率 |
| 辅助指标 | F1-Score | 精确率与召回率的调和平均 |
| 效率指标 | FPS | 每秒推理帧数 |
| 效率指标 | 参数量(M) | 模型参数数量 |
| 效率指标 | FLOPs(G) | 浮点运算次数 |
| 效率指标 | 模型大小(MB) | 模型文件大小 |

### 6.3 消融实验设计

| 组件 | E1 | E2 | E3 | E4 | E5 | E6 |
|------|----|----|----|----|----|----|
| 数据增强 | 基础 | 增强 | 增强 | 增强 | 增强 | 增强 |
| CA注意力 | - | - | ✅ | ✅ | ✅ | ✅ |
| BiFPN | - | - | - | ✅ | ✅ | ✅ |
| SAHI | - | - | - | - | ✅ | ✅ |
| YOLOv11s | - | - | - | - | - | ✅ |

---

## 7. 项目结构与规范

### 7.1 目录结构

```
yolo-wind-turbine/
├── README.md                    # 项目说明
├── PROJECT_PLAN.md              # 详细项目计划
├── TECHNICAL_REPORT.md          # 技术报告(本文档)
├── requirements.txt             # Python依赖
├── setup_env.bat                # 一键环境配置
├── data/
│   ├── wind_turbine.yaml        # 数据集配置
│   ├── images/{train,val,test}/ # 图片目录
│   ├── labels/{train,val,test}/ # YOLO标注
│   └── raw/                     # 原始数据集
├── src/
│   ├── train.py                 # 基线训练
│   ├── train_improved.py        # 改进训练
│   ├── detect.py                # 推理脚本
│   ├── evaluate.py              # 评估脚本
│   └── utils.py                 # 工具函数
├── configs/
│   ├── yolov11n.yaml            # YOLOv11-nano
│   └── yolov11s.yaml            # YOLOv11-small+CA
├── scripts/
│   ├── download_datasets.py     # 数据集下载
│   ├── merge_datasets.py        # 数据合并
│   └── split_dataset.py         # 数据划分
├── docs/
│   ├── literature_review.md     # 文献综述
│   └── experiment_log.md        # 实验记录
├── results/                     # 实验结果
└── runs/                        # 训练权重
```

### 7.2 环境配置

```bash
# 激活环境
conda activate yolo-project

# 安装依赖
pip install -r requirements.txt

# 验证
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

### 7.3 训练命令

```bash
# 基线训练
python src/train.py --model yolo11n.pt --epochs 100 --batch 16

# 改进训练
python src/train_improved.py --model yolo11n.pt --use-ca --epochs 100

# 评估
python src/evaluate.py --weights runs/train/baseline/weights/best.pt

# 推理
python src/detect.py --weights runs/train/baseline/weights/best.pt --source test_images/
```

---

## 8. 参考文献

1. Zhang & Wen, "SOD-YOLO: Small Object Detection YOLO Based on Improved YOLOv5", 2022
2. Tong et al., "WTBD-YOLOv8: Wind Turbine Blade Defect Detection", 2024
3. Fu et al., "LE-YOLO: Enhanced YOLO for Leading Edge Defect Detection", 2024
4. Zhang et al., "GCB-YOLO: GhostNet+CA+BiFPN for Blade Detection", 2025
5. Memari et al., "Wind Turbine Blade Defect Detection with YOLO Models", 2024 (114引用)
6. Masita et al., "Deep Learning for WTB Defect Detection: A Review", 2025
7. Wang et al., "Lightning Strike Damage Detection for Wind Turbines", 2022 (108引用)
8. Rabbi et al., "VR-generated Synthetic Data for Blade Defect Detection", 2023
9. Liu et al., "Attention-mechanism-based Surface Defect Detection", 2023 (65引用)
10. Yang et al., "Blade30: Wind Turbine Blade Dataset", Renewable Energy, 2023

---

*文档编制: Claude Code | 日期: 2026-05-18*
