# 项目使用教程

> 从零开始复现 YOLO + VisDrone 低空小目标检测项目
> 适用平台: Windows 10/11, Linux (Ubuntu 20.04+)

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据集下载与处理](#2-数据集下载与处理)
3. [训练模型](#3-训练模型)
4. [模型评估](#4-模型评估)
5. [推理检测](#5-推理检测)
6. [SAHI 切片推理](#6-sahi-切片推理)
7. [零训练优化](#7-零训练优化)
8. [可视化](#8-可视化)
9. [常见问题 FAQ](#9-常见问题-faq)

---

## 1. 环境准备

### 1.1 硬件要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GTX 1060 6GB | RTX 4060 8GB+ |
| 内存 | 8GB | 16GB+ |
| 硬盘 | 15GB 可用空间 | 30GB+ |
| Python | 3.8+ | 3.9 |

### 1.2 一键配置 (Windows)

双击运行 `setup_env.bat`，自动完成:
1. 检测 Python 版本
2. 检测 NVIDIA GPU
3. 安装所有依赖
4. 验证 PyTorch + CUDA

```
========================================
  YOLO + VisDrone 环境配置
========================================

[OK] 检测到 NVIDIA GPU
NVIDIA GeForce RTX 4060 Laptop, 8192 MiB

[1/3] 安装 Python 依赖...
[2/3] 验证 Ultralytics 安装...
PyTorch: 2.6.0+cu124
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Laptop

[3/3] 配置完成！
```

### 1.3 手动配置

```bash
# 1. 创建虚拟环境 (推荐)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# 2. 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "from ultralytics import YOLO; import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 1.4 依赖清单

| 包 | 版本 | 用途 |
|---|------|------|
| ultralytics | >=8.0.0 | YOLOv8 框架 |
| torch | >=2.0.0 | 深度学习框架 |
| opencv-python | >=4.8.0 | 图像处理 |
| matplotlib | >=3.7.0 | 绘图 |
| pandas | >=2.0.0 | 数据分析 |
| seaborn | >=0.12.0 | 统计可视化 |
| tqdm | >=4.65.0 | 进度条 |
| numpy | >=1.24.0 | 数值计算 |
| sahi | >=0.11.0 | 切片推理 |
| pycocotools | - | COCO 评估 |

---

## 2. 数据集下载与处理

### 2.1 一键下载

```bash
python download_data.py
```

该脚本自动完成:
1. 从 GitHub 下载 VisDrone2019-DET 数据集 (train/val/test)
2. 解压到 `data/visdrone/VisDrone2019-DET-{split}/`
3. 转换标注格式 (VisDrone → YOLO)
4. 整理到 `data/visdrone/images/` 和 `data/visdrone/labels/`

### 2.2 VisDrone 标注格式转换

**原始格式** (VisDrone):
```
<left>,<top>,<width>,<height>,<score>,<object_category>,<truncation>,<occlusion>
```
示例: `543,302,45,67,1,0,0,0`

**目标格式** (YOLO):
```
<class_id> <x_center> <y_center> <width> <height>
```
示例: `0 0.422 0.481 0.033 0.088`

**转换公式**:
```python
# VisDrone: (left, top, width, height) 像素坐标
# YOLO: (x_center, y_center, width, height) 归一化坐标 [0, 1]

x_center = (left + width / 2) / image_width
y_center = (top + height / 2) / image_height
norm_width = width / image_width
norm_height = height / image_height
```

**过滤规则**:
- 忽略 class=0 (ignored regions)
- 过滤 score < 0 的标注

### 2.3 数据集结构

```
data/visdrone/
├── visdrone.yaml           # YOLO 数据集配置文件
├── images/
│   ├── train/              # 6,471 张训练图片
│   └── val/                # 548 张验证图片
└── labels/
    ├── train/              # 6,471 个训练标注
    └── val/                # 548 个验证标注
```

### 2.4 数据集配置文件

`data/visdrone/visdrone.yaml`:
```yaml
path: data/visdrone
train: images/train
val: images/val
test: images/test

nc: 10
names:
  0: pedestrian    # 行人
  1: people        # 人群
  2: bicycle       # 自行车
  3: car           # 汽车
  4: van           # 面包车
  5: truck         # 卡车
  6: tricycle      # 三轮车
  7: awning-tricycle  # 篷三轮车
  8: bus           # 公交车
  9: motor         # 摩托车
```

---

## 3. 训练模型

### 3.1 基线模型训练

```bash
python src/train.py --model yolov8n --epochs 50 --imgsz 640 --batch 16 --name baseline --exp yolov8n_visdrone
```

### 3.2 改进模型训练

```bash
python src/train.py --model yolov8s --epochs 80 --imgsz 800 --batch 8 --name improved --exp yolov8s_visdrone
```

### 3.3 参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | yolov8n | 模型变体: yolov8n/s/m/l/x |
| `--weights` | None | 自定义权重路径 (优先级高于 --model) |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 640 | 输入图片尺寸 |
| `--batch` | 16 | 批量大小 (显存不足时减小) |
| `--workers` | 4 | 数据加载线程数 |
| `--device` | 0 | GPU 设备号 (CPU 用 "cpu") |
| `--name` | baseline | 实验类别目录 |
| `--exp` | yolov8n_visdrone | 实验名称 |

### 3.4 模型变体对比

| 模型 | 参数量 | GFLOPs | COCO mAP | 适用场景 |
|------|--------|--------|----------|---------|
| YOLOv8n | 3.2M | 8.7 | 37.3 | 边缘设备、实时检测 |
| YOLOv8s | 11.2M | 28.6 | 44.9 | 平衡速度和精度 |
| YOLOv8m | 25.9M | 78.9 | 50.2 | 高精度需求 |
| YOLOv8l | 43.7M | 165.2 | 52.9 | 大规模部署 |
| YOLOv8x | 68.2M | 257.8 | 53.9 | 最高精度 |

### 3.5 训练超参数

本项目使用的训练超参数 (在 `src/train.py` 中硬编码):

```python
optimizer = "auto"        # 自动选择优化器 (SGD/Adam/AdamW)
lr0 = 0.01               # 初始学习率
lrf = 0.01               # 最终学习率 = lr0 × lrf
momentum = 0.937          # SGD 动量
weight_decay = 0.0005     # 权重衰减
warmup_epochs = 3         # 预热轮数
warmup_momentum = 0.8     # 预热动量
warmup_bias_lr = 0.1      # 预热偏置学习率
box = 7.5                 # 边界框损失权重
cls = 0.5                 # 分类损失权重
dfl = 1.5                 # DFL 损失权重
close_mosaic = 10         # 最后 N 轮关闭 Mosaic
amp = True                # 混合精度训练
```

### 3.6 训练监控

使用 `monitor_training.py` 实时监控训练进度和磁盘空间:

```bash
python monitor_training.py
```

功能:
- 每 2 分钟检查 C: 和 E: 盘剩余空间
- C: < 5GB 或 E: < 10GB 时报警
- 读取 `results.csv` 显示当前 epoch 和 mAP
- 检测训练完成 (best.pt 出现)

### 3.7 训练输出

训练完成后，结果保存在 `runs/<name>/<exp>/`:

```
runs/baseline/yolov8n_visdrone/
├── weights/
│   ├── best.pt              # 最佳模型权重
│   └── last.pt              # 最后一轮权重
├── results.csv              # 每个 epoch 的指标
├── results.png              # 训练曲线图
├── args.yaml                # 训练参数记录
├── confusion_matrix.png     # 混淆矩阵
├── BoxF1_curve.png          # F1 曲线
├── BoxPR_curve.png          # PR 曲线
├── BoxP_curve.png           # Precision 曲线
├── BoxR_curve.png           # Recall 曲线
├── labels.jpg               # 标签分布图
├── train_batch0.jpg         # 训练批次样本
└── val_batch0_pred.jpg      # 验证集预测结果
```

---

## 4. 模型评估

### 4.1 基本评估

```bash
# 评估基线模型
python src/evaluate.py --weights runs/baseline/yolov8n_visdrone/weights/best.pt --name baseline_eval

# 评估改进模型
python src/evaluate.py --weights runs/improved/yolov8s_visdrone/weights/best.pt --name improved_eval
```

### 4.2 输出指标

```
==================================================
  评估结果
==================================================
  mAP@0.5:      0.4258
  mAP@0.5:0.95: 0.2453
  Precision:    0.5549
  Recall:       0.4244

  各类别 mAP@0.5:
    pedestrian        : 0.4789
    people            : 0.3505
    bicycle           : 0.1756
    car               : 0.8149
    van               : 0.4784
    truck             : 0.4050
    tricycle          : 0.3075
    awning-tricycle   : 0.1712
    bus               : 0.5892
    motor             : 0.4867
```

### 4.3 SAHI 评估

```bash
python src/sahi_eval.py
```

使用 COCO 格式评估 SAHI 切片推理效果，输出 `results/sahi_metrics.json`。

---

## 5. 推理检测

### 5.1 单张图片推理

```bash
python src/detect.py \
    --weights runs/improved/yolov8s_visdrone/weights/best.pt \
    --source data/visdrone/images/val/0000001_02999_d_0000005.jpg \
    --conf 0.25
```

### 5.2 批量推理

```bash
# 对整个验证集推理
python src/detect.py \
    --weights runs/improved/yolov8s_visdrone/weights/best.pt \
    --source data/visdrone/images/val/ \
    --name improved_detect
```

### 5.3 视频推理

```bash
python src/detect.py \
    --weights runs/improved/yolov8s_visdrone/weights/best.pt \
    --source video.mp4 \
    --conf 0.3
```

### 5.4 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | (必填) | 模型权重路径 |
| `--source` | (必填) | 输入源: 图片/视频/目录/摄像头(0) |
| `--imgsz` | 640 | 推理尺寸 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.45 | NMS IoU 阈值 |
| `--device` | 0 | GPU 设备号 |
| `--name` | exp | 输出目录名 |

---

## 6. SAHI 切片推理

### 6.1 原理

SAHI 将大图切成重叠小块，分别用 YOLO 检测，再 NMS 合并结果:

```
原图 (1360×765) → 切片 (640×640, 重叠20%) → 逐片推理 → NMS 合并 → 最终结果
```

小目标在切片中像素占比从 ~1% 提升到 ~5%，检测精度显著提升。

### 6.2 SAHI 评估

```bash
python src/sahi_eval.py
```

输出 `results/sahi_metrics.json`，包含 mAP 和各类别 AP。

### 6.3 SAHI 检测可视化

```bash
# 对 10 张图片生成 SAHI vs 普通推理对比图
python src/sahi_detect.py --max 10

# 指定单张图片
python src/sahi_detect.py --source data/visdrone/images/val/0000001_02999_d_0000005.jpg
```

输出到 `results/sahi_compare/`。

### 6.4 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| slice_height/width | 640 | 切片尺寸 (匹配模型训练尺寸) |
| overlap_ratio | 0.2 | 重叠率 (防止边界截断) |
| conf_threshold | 0.25 | 置信度阈值 |
| postprocess_type | NMS | 后处理方式 |
| match_threshold | 0.5 | NMS 匹配阈值 |

---

## 7. 零训练优化

### 7.1 运行全部优化实验

```bash
python src/optimize.py
```

该脚本依次执行三个实验:
1. **SAHI + TTA**: 水平翻转增强，per-class NMS 合并
2. **阈值网格搜索**: 在 conf=0.05~0.40 间搜索最优阈值
3. **多模型集成**: YOLOv8n (×0.7) + YOLOv8s (×1.0) 加权融合

### 7.2 输出文件

```
results/
├── sahi_tta_metrics.json        # SAHI+TTA 结果
├── threshold_search.json        # 阈值搜索结果 (含网格搜索明细)
├── ensemble_metrics.json        # 集成结果
└── optimization_summary.json    # 综合排名
```

### 7.3 结果解读

```json
{
  "all_results": [
    {"method": "SAHI(conf=0.05)", "mAP50": 0.4903},
    {"method": "SAHI+TTA", "mAP50": 0.4710},
    {"method": "sahi", "mAP50": 0.4523},
    {"method": "improved", "mAP50": 0.4258},
    {"method": "Ensemble", "mAP50": 0.3535},
    {"method": "baseline", "mAP50": 0.2979}
  ],
  "best_method": "SAHI(conf=0.05)",
  "best_mAP50": 0.4903
}
```

---

## 8. 可视化

### 8.1 检测结果图

```bash
# 生成检测结果可视化
python src/visualize.py detect \
    --weights runs/improved/yolov8s_visdrone/weights/best.pt \
    --source data/visdrone/images/val \
    --output results/improved_detections \
    --max 15
```

### 8.2 多模型对比图

```bash
# 基线 vs 改进模型并排对比
python src/visualize.py compare \
    --models baseline=runs/baseline/yolov8n_visdrone/weights/best.pt improved=runs/improved/yolov8s_visdrone/weights/best.pt \
    --source data/visdrone/images/val \
    --output results/compare \
    --max 10
```

### 8.3 热力图

```bash
python src/visualize.py heatmap \
    --weights runs/improved/yolov8s_visdrone/weights/best.pt \
    --source data/visdrone/images/val/0000001_02999_d_0000005.jpg \
    --output results/heatmap.jpg
```

---

## 9. 常见问题 FAQ

### Q1: CUDA out of memory

**原因**: 显存不足

**解决**:
```bash
# 减小 batch size
python src/train.py --batch 8   # 或 4

# 减小输入尺寸
python src/train.py --imgsz 640

# 关闭其他占用显存的程序
```

### Q2: 训练时 mAP 不收敛

**可能原因**:
- 学习率过大 → 降低 `lr0`
- 数据增强过强 → 降低增强参数
- 标注有误 → 检查标签文件

### Q3: pip install 报代理错误

```bash
# 清除代理环境变量
# Windows PowerShell:
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""

# 然后重新安装
pip install -r requirements.txt
```

### Q4: pi_heif 模块找不到

**原因**: ultralytics 尝试导入 pi_heif，但 Windows 上无法编译

**解决**: 项目代码已处理此问题，在导入前保存并恢复 `PIL.Image.open`:
```python
import PIL.Image
_pil_open = PIL.Image.open
# ... import ultralytics ...
PIL.Image.open = _pil_open
```

### Q5: VisDrone 测试集下载 404

**原因**: 官方链接已失效

**解决**: 项目使用 GitHub Ultralytics 镜像下载，如果仍然失败，可手动从 [VisDrone 官网](http://aiskyeye.com/) 下载。

### Q6: 如何使用自己的数据集？

1. 准备 YOLO 格式标注 (`<class> <x_center> <y_center> <w> <h>`)
2. 创建 `your_dataset.yaml`:
   ```yaml
   path: data/your_dataset
   train: images/train
   val: images/val
   nc: 你的类别数
   names: ['class0', 'class1', ...]
   ```
3. 训练:
   ```bash
   python src/train.py --model yolov8n --weights yolov8n.pt
   ```
   修改 `src/train.py` 中的 `data_yaml` 路径指向你的 yaml 文件。

---

## 附录: 完整实验复现步骤

```bash
# 1. 环境配置
setup_env.bat  # 或手动 pip install -r requirements.txt

# 2. 下载数据集
python download_data.py

# 3. 训练基线模型 (约 65 分钟)
python src/train.py --model yolov8n --epochs 50 --imgsz 640 --batch 16 --name baseline --exp yolov8n_visdrone

# 4. 训练改进模型 (约 2.5 小时)
python src/train.py --model yolov8s --epochs 80 --imgsz 800 --batch 8 --name improved --exp yolov8s_visdrone

# 5. 评估模型
python src/evaluate.py --weights runs/baseline/yolov8n_visdrone/weights/best.pt --name baseline_eval
python src/evaluate.py --weights runs/improved/yolov8s_visdrone/weights/best.pt --name improved_eval

# 6. SAHI 切片推理评估 (约 3 分钟)
python src/sahi_eval.py

# 7. 零训练优化实验 (约 25 分钟)
python src/optimize.py

# 8. 生成可视化
python src/visualize.py detect --weights runs/improved/yolov8s_visdrone/weights/best.pt --source data/visdrone/images/val --max 15
python src/sahi_detect.py --max 10
```
