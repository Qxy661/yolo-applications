# 数据集盘点与整理方案

> 盘点时间: 2026-05-23
> 数据目录: E:\yolo-wind-turbine\data

---

## 一、当前数据盘点

### 1.1 已有数据集

| 数据集 | 目录 | 图片数 | 标注格式 | 标注数 | 类别数 | 状态 |
|--------|------|--------|----------|--------|--------|------|
| Blade30 叶片1-15 | `3_blade_1_15_with_labeldata/` | 605 | JSON (LabelMe) | ~106 | 5类 | ✅ 已下载 |
| Blade30 叶片16-30 | `3_blade_16_30_with_labeldata/` | 697 | JSON (LabelMe) | ~106 | 5类 | ✅ 已下载 |
| WT blade defect | `WT blade defect dataset/` | 1,065 | VOC XML | 1,065 | 6类 | ✅ 已下载 |
| **合计** | | **2,367** | | | | |

### 1.2 Blade30 数据集详情

**目录结构**:
```
3_blade_1_15_with_labeldata/
└── 3_blade_1_15_with_labeldata/
    ├── Blade_1/
    │   └── 1_1/
    │       ├── *.jpg          # 图片
    │       ├── *.json         # LabelMe标注
    │       └── mask/
    │           └── *.png      # 分割掩码
    ├── Blade_2/
    │   └── ...
    └── Blade_15/

3_blade_16_30_with_labeldata/
└── 3_blade_16_30_with_labeldata/
    ├── Blade_16/
    │   └── 2_1/
    │       ├── *.jpg
    │       ├── *.json
    │       └── mask/
    └── Blade_30/
```

**类别** (5类):
- crack (裂纹)
- erosion (侵蚀)
- lightning (雷击)
- peeling (涂层脱落)
- hole (孔洞)

**标注格式** (JSON/LabelMe):
```json
{
  "shapes": [
    {
      "label": "crack",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

### 1.3 WT blade defect dataset 详情

**目录结构**:
```
WT blade defect dataset/
└── WT blade defect dataset/
    ├── JPEGImages/           # 1,065张图片
    │   ├── 0.jpg
    │   └── ...
    ├── Annotations/          # 1,065个VOC XML标注
    │   ├── 0.xml
    │   └── ...
    ├── train_val_test_split.txt  # 数据划分
    ├── class_definitions.txt     # 类别定义
    └── *.py                      # 辅助脚本
```

**类别** (6类):
1. craze (裂纹)
2. corrosion (腐蚀)
3. surface_injure (表面损伤)
4. thunderstrike (雷击)
5. crack (裂纹)
6. hide_craze (隐藏裂纹)

**标注格式** (VOC XML):
```xml
<annotation>
  <filename>0.jpg</filename>
  <size>
    <width>1024</width>
    <height>1024</height>
  </size>
  <object>
    <name>corrosion</name>
    <bndbox>
      <xmin>304</xmin>
      <ymin>13</ymin>
      <xmax>848</xmax>
      <ymax>1020</ymax>
    </bndbox>
  </object>
</annotation>
```

**数据划分**:
- train: 约70%
- val: 约15%
- test: 约15%

---

## 二、类别映射方案

### 2.1 标准5类定义

| ID | 类别 | 英文 | 中文 | 描述 |
|----|------|------|------|------|
| 0 | crack | Crack | 裂纹 | 叶片表面裂纹 |
| 1 | erosion | Erosion | 侵蚀 | 表面侵蚀、腐蚀、脏污 |
| 2 | lightning | Lightning | 雷击 | 雷击损伤 |
| 3 | peeling | Peeling | 涂层脱落 | 涂层/材料脱落 |
| 4 | hole | Hole | 孔洞 | 针孔或较大孔洞 |

### 2.2 Blade30 类别映射 (直接对应)

| Blade30类别 | → | 标准类别 |
|-------------|---|----------|
| crack | → | crack (0) |
| erosion | → | erosion (1) |
| lightning | → | lightning (2) |
| peeling | → | peeling (3) |
| hole | → | hole (4) |

### 2.3 WT blade defect 类别映射

| WT类别 | → | 标准类别 | 说明 |
|--------|---|----------|------|
| craze | → | crack (0) | 裂纹类 |
| crack | → | crack (0) | 裂纹类 |
| hide_craze | → | crack (0) | 隐藏裂纹 |
| corrosion | → | erosion (1) | 腐蚀类 |
| surface_injure | → | erosion (1) | 表面损伤 |
| thunderstrike | → | lightning (2) | 雷击 |
| (无对应) | → | peeling (3) | 缺少此类 |
| (无对应) | → | hole (4) | 缺少此类 |

---

## 三、整理方案

### 3.1 目标目录结构

```
data/
├── raw/                              # 原始数据 (保持不动)
│   ├── blade30_1_15/                 # Blade30 叶片1-15
│   ├── blade30_16_30/                # Blade30 叶片16-30
│   └── wt_blade_defect/              # WT blade defect dataset
├── processed/                        # 处理后数据
│   └── detection/                    # 目标检测格式
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── scripts/                          # 处理脚本
│   ├── 01_organize_data.py           # 数据整理
│   ├── 02_convert_format.py          # 格式转换
│   ├── 03_merge_datasets.py          # 合并数据集
│   ├── 04_split_dataset.py           # 划分数据集
│   └── 05_eda_report.py              # EDA分析
├── docs/                             # 文档
│   ├── data_inventory.md             # 本文档
│   ├── processing_log.md             # 处理日志
│   └── eda_report.md                 # EDA报告
└── wind_turbine.yaml                 # YOLO配置文件
```

### 3.2 处理流程

```
Step 1: 数据整理 (01_organize_data.py)
  ├── 扫描 raw/ 目录
  ├── 统计各数据集信息
  └── 生成数据清单

Step 2: 格式转换 (02_convert_format.py)
  ├── Blade30: JSON → YOLO txt
  ├── WT dataset: VOC XML → YOLO txt
  └── 类别映射到标准5类

Step 3: 合并数据集 (03_merge_datasets.py)
  ├── 合并所有数据集
  ├── 基于图片哈希去重
  └── 统一类别标注

Step 4: 划分数据集 (04_split_dataset.py)
  ├── 分层抽样划分 7:1:2
  ├── 确保类别分布一致
  └── 复制到标准目录

Step 5: EDA分析 (05_eda_report.py)
  ├── 类别分布统计
  ├── 图片尺寸分析
  ├── 标注框大小分布
  └── 生成可视化报告
```

---

## 四、预期结果

### 4.1 数据量预估

| 数据集 | 图片数 | 标注数 | 有效标注 |
|--------|--------|--------|----------|
| Blade30 叶片1-15 | 605 | ~106 | ~106 |
| Blade30 叶片16-30 | 697 | ~106 | ~106 |
| WT blade defect | 1,065 | 1,065 | ~800+ |
| **合计** | **2,367** | | |

### 4.2 类别分布预估

| 类别 | Blade30 | WT dataset | 总计 |
|------|---------|------------|------|
| crack | ~30 | ~300+ | ~330+ |
| erosion | ~30 | ~200+ | ~230+ |
| lightning | ~20 | ~100+ | ~120+ |
| peeling | ~20 | 0 | ~20+ |
| hole | ~10 | 0 | ~10+ |

### 4.3 数据集划分

| 子集 | 比例 | 预估数量 |
|------|------|----------|
| train | 70% | ~1,657 |
| val | 15% | ~355 |
| test | 15% | ~355 |

---

## 五、待办事项

### 优先级 1: 立即执行

- [ ] 运行数据整理脚本，生成完整数据清单
- [ ] 转换Blade30 JSON标注为YOLO格式
- [ ] 转换WT dataset VOC XML为YOLO格式
- [ ] 合并两个数据集

### 优先级 2: 后续执行

- [ ] 划分训练/验证/测试集
- [ ] 运行EDA分析
- [ ] 生成数据处理报告

### 优先级 3: 可选执行

- [ ] 下载更多数据集 (Kaggle, QQ系列)
- [ ] 数据增强 (Mosaic, MixUp)
- [ ] 训练验证

---

## 六、注意事项

1. **保留原始数据**: raw/ 目录保持不动，所有处理结果放到 processed/
2. **备份重要文件**: 处理前备份原始标注
3. **日志记录**: 每步操作都记录到 processing_log.md
4. **验证结果**: 每步完成后验证输出是否正确
5. **Git管理**: 重要节点提交Git，方便回溯

---

## 七、参考资源

- Blade30论文: Yang et al., "Blade30", Renewable Energy, 2023
- YOLO格式: https://docs.ultralytics.com/datasets/detect/
- LabelMe格式: https://github.com/labelmeai/labelme
- VOC格式: http://host.robots.ox.ac.uk/pascal/VOC/
