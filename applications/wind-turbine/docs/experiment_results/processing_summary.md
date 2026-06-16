# 数据集处理总结

> 处理时间: 2026-05-23
> 数据目录: E:\yolo-wind-turbine\data

---

## 一、处理结果概览

### 1.1 数据集统计

| 数据集 | 原始图片 | 处理后 | 标注数 | 状态 |
|--------|----------|--------|--------|------|
| Blade30 叶片1-15 | 605 | 57 | 57 | ✅ |
| Blade30 叶片16-30 | 697 | 36 | 36 | ✅ |
| WT blade defect | 1,065 | 1,065 | 1,065 | ✅ |
| **合计** | **2,367** | **1,158** | **1,158** | |

### 1.2 最终数据集

| 子集 | 图片数 | 标注数 | 比例 |
|------|--------|--------|------|
| train | 810 | 810 | 70% |
| val | 172 | 172 | 15% |
| test | 176 | 176 | 15% |
| **合计** | **1,158** | **1,158** | 100% |

### 1.3 类别分布

| 类别 | 数量 | 占比 | 状态 |
|------|------|------|------|
| crack | 832 | 47.4% | ⚠️ 过多 |
| erosion | 833 | 47.4% | ⚠️ 过多 |
| lightning | 92 | 5.2% | ❌ 不足 |
| peeling | 0 | 0% | ❌ 缺失 |
| hole | 0 | 0% | ❌ 缺失 |

---

## 二、处理流程

### 2.1 数据转换

**Blade30 数据集**:
- 输入: JSON (LabelMe格式)
- 输出: YOLO txt格式
- 类别映射:
  - `trailing edge;crack;*` → crack (0)
  - `leading edge;erosion;*` → erosion (1)
  - `surface;contamination;*` → erosion (1)
  - `*;lightning;*` → lightning (2)

**WT blade defect 数据集**:
- 输入: VOC XML格式
- 输出: YOLO txt格式
- 类别映射:
  - `craze`, `crack`, `hide_craze` → crack (0)
  - `corrosion`, `surface_injure` → erosion (1)
  - `thunderstrike` → lightning (2)

### 2.2 数据划分

**Blade30 数据集**:
- 原始划分: 无
- 新划分: 70% train, 15% val, 15% test
- 方法: 随机分层抽样

**WT blade defect 数据集**:
- 原始划分: train=745, val=159, test=161
- 保持原始划分不变

### 2.3 数据合并

- 合并两个数据集到 `processed/detection/`
- 目录结构: images/{train,val,test} + labels/{train,val,test}
- 格式: YOLO txt (class_id cx cy w h)

---

## 三、关键发现

### 3.1 数据质量问题

1. **Blade30 标注覆盖率低**: 605张图片只有93张有有效标注 (15.4%)
2. **类别不平衡严重**:
   - crack + erosion 占 94.8%
   - lightning 仅占 5.2%
   - peeling 和 hole 完全缺失

2. **标注格式复杂**: Blade30使用分号分隔的多级标签，需要特殊处理

### 3.2 数据增强建议

由于类别不平衡严重，建议:

1. **使用加权损失函数**:
   - lightning: 权重 ×5
   - peeling: 权重 ×10 (如后续补充数据)
   - hole: 权重 ×10 (如后续补充数据)

2. **数据增强**:
   - Mosaic: +30% mAP
   - MixUp: +10% mAP
   - CopyPaste: 针对少数类别

3. **分辨率渐进训练**:
   - Phase 1: 640px
   - Phase 2: 1280px

---

## 四、目录结构

```
data/
├── raw/                              # 原始数据 (保持不动)
│   ├── 3_blade_1_15_with_labeldata/
│   ├── 3_blade_16_30_with_labeldata/
│   └── WT blade defect dataset/
├── processed/                        # 处理后数据
│   └── detection/
│       ├── images/
│       │   ├── train/               # 810张
│       │   ├── val/                 # 172张
│       │   └── test/                # 176张
│       └── labels/
│           ├── train/               # 810个
│           ├── val/                 # 172个
│           └── test/                # 176个
├── scripts/
│   ├── 01_organize_data.py          # 数据盘点
│   ├── 02_process_all.py            # 完整处理流程
│   └── ...
├── docs/
│   ├── data_inventory.md            # 数据盘点文档
│   ├── processing_summary.md        # 本文档
│   ├── eda_report.md                # EDA报告
│   └── inventory.json               # 数据清单
└── wind_turbine.yaml                # YOLO配置文件
```

---

## 五、训练建议

### 5.1 模型配置

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='data/wind_turbine.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    cls_pw=2.0,  # 类别权重，解决不平衡
)
```

### 5.2 预期结果

| 指标 | 预期 | 说明 |
|------|------|------|
| mAP@0.5 | 70-85% | 受限于数据量和类别不平衡 |
| mAP@0.5:0.95 | 50-65% | 严格指标 |
| FPS | >60 | YOLOv11n轻量级 |
| 模型大小 | <10MB | 适合部署 |

### 5.3 改进方向

1. **补充数据**:
   - 下载更多 peeling 和 hole 类别数据
   - 搜索 Kaggle、AI Studio 等平台

2. **数据增强**:
   - 使用 CopyPaste 增强少数类别
   - 使用 Mosaic + MixUp

3. **模型优化**:
   - 使用 GhostNet 轻量化
   - 添加 CA 注意力机制
   - 使用 BiFPN 特征融合

---

## 六、参考资源

- YOLOv11: https://docs.ultralytics.com/
- Blade30: https://github.com/cong-yang/Blade30
- 数据增强: https://docs.ultralytics.com/modes/train/#augmentation-settings
