# 当前数据集状态分析

> 分析时间: 2026-05-23
> 项目: 风电叶片缺陷检测

## 一、当前状态概览

| 数据集 | 目录 | 状态 | 数据量 |
|--------|------|------|--------|
| Blade30 | `raw/blade30/` | ❌ 空目录 | 0 张 |
| QQ767172261 六类 | `raw/qq767172261_6cls/repo/` | ⚠️ 仅有README | 0 张 |
| QQ767172261 UAV五类 | `raw/qq767172261_uav5/repo/` | ⚠️ 仅有README | 0 张 |
| WTBD | `raw/wtbd/` | ❌ 空目录 | 0 张 |

**结论：所有数据集均未下载完成，需要手动下载。**

---

## 二、各数据集详细分析

### 2.1 Blade30 数据集

| 项目 | 信息 |
|------|------|
| **来源** | Renewable Energy, Vol.203, 2023 |
| **论文** | https://github.com/cong-yang/Blade30 |
| **图片数量** | 1,302 张 |
| **叶片数量** | 30 片 |
| **标注格式** | JSON (LabelMe格式) |
| **类别** | crack, erosion, lightning, peeling, hole |
| **特点** | 真实无人机巡检数据，质量最高 |

**下载方式**:
- 百度网盘 (推荐):
  - 叶片1-15: `https://pan.baidu.com/s/17kv5Xadz1QcSrvoG58WtBw` (码:1234)
  - 叶片16-30: `https://pan.baidu.com/s/1hzcwdc6sBXOeja3nkfartg` (码:1234)
- Google Drive:
  - 叶片1-15: `https://drive.google.com/uc?id=1HbB4t9xV2oCgSSxR9hMEOU6v9qDfetmR`
  - 叶片16-30: `https://drive.google.com/uc?id=1SwRdMzA7zCkNVlHuWvk8uK6eDToM0mUV`
- OneDrive: `https://1drv.ms/u/s!AoXJBmXKVWu5tmtUzCJULhrtYuIP`

**预期目录结构**:
```
blade30/
├── images/
│   ├── blade_001/
│   │   ├── blade_001_001.jpg
│   │   └── ...
│   └── ...
└── annotations/
    ├── blade_001.json
    └── ...
```

---

### 2.2 QQ767172261 六类数据集

| 项目 | 信息 |
|------|------|
| **来源** | GitHub (QQ767172261) |
| **图片数量** | 3,282 张 |
| **标注格式** | YOLO txt |
| **数据划分** | 训练集: 2,743 / 验证集: 270 / 测试集: 269 |

**类别分布**:

| 类别 | 英文名 | 图片数 | 标注数 | 映射到标准类别 |
|------|--------|--------|--------|----------------|
| 裂纹 | Crack | 1,144 | 2,163 | → crack |
| 侵蚀 | Erosion | 233 | 337 | → erosion |
| 脏污 | Dirt | 693 | 762 | → erosion |
| 漏油 | Oil leakage | 574 | 600 | → erosion |
| PU胶带 | PU-tape | 329 | 621 | → peeling |
| 孔洞 | Pin Hole | 303 | 303 | → hole |

**类别映射问题**:
- 原始6类 → 标准5类
- `Dirt` 和 `Oil leakage` 需要合并到 `erosion`
- `PU-tape` 映射到 `peeling`
- 缺少 `lightning` 类别

---

### 2.3 QQ767172261 UAV五类数据集

| 项目 | 信息 |
|------|------|
| **来源** | GitHub (QQ767172261) |
| **图片数量** | 4,467 张 |
| **标注格式** | YOLO txt |
| **数据划分** | 训练集: 3,898 / 验证集: 189 / 测试集: 380 |

**类别分布**:

| 类别ID | 英文名 | 中文名 | 样本数 | 映射到标准类别 |
|--------|--------|--------|--------|----------------|
| 0 | OIL LEAKAGE | 油渍 | 753 | → erosion |
| 1 | dirt | 污垢 | 846 | → erosion |
| 2 | Paint | 剥落 | 2,455 | → peeling |
| 3 | LE-Erosion | 侵蚀 | 617 | → erosion |
| 4 | PU-tape | PU胶带 | 700 | → peeling |

**类别映射问题**:
- 原始5类 → 标准5类
- `OIL LEAKAGE`、`dirt`、`LE-Erosion` 都映射到 `erosion`
- `Paint` 和 `PU-tape` 映射到 `peeling`
- 缺少 `crack`、`lightning`、`hole` 类别

**训练结果参考** (YOLOv8n):
- mAP@0.5: 84.2%
- 适合边缘部署

---

### 2.4 WTBD 数据集

| 项目 | 信息 |
|------|------|
| **来源** | Nature Scientific Data 2026 |
| **论文** | https://www.nature.com/articles/s41597-026-06762-x |
| **状态** | ❌ 未获取 |

**获取方式**:
1. 访问论文页面
2. 查找 Data Availability 或 Data Citations 部分
3. 常见下载平台: figshare.com, zenodo.org, GitHub

---

## 三、类别映射汇总

### 标准5类定义

| ID | 类别 | 英文 | 中文 | 描述 |
|----|------|------|------|------|
| 0 | crack | Crack | 裂纹 | 叶片表面裂纹 |
| 1 | erosion | Erosion | 侵蚀 | 表面侵蚀、脏污、漏油 |
| 2 | lightning | Lightning | 雷击 | 雷击损伤 |
| 3 | peeling | Peeling | 涂层脱落 | 涂层/材料脱落 |
| 4 | hole | Hole | 孔洞 | 针孔或较大孔洞 |

### 各数据集映射关系

```
Blade30 (5类) ─────────────────────────────→ 标准5类 (直接对应)

QQ767172261 六类 (6类) ───────────────────→ 标准5类
  ├── Crack ──────────────────────────────→ crack
  ├── Erosion ────────────────────────────→ erosion
  ├── Dirt ───────────────────────────────→ erosion (合并)
  ├── Oil leakage ────────────────────────→ erosion (合并)
  ├── PU-tape ────────────────────────────→ peeling
  └── Pin Hole ───────────────────────────→ hole

QQ767172261 UAV五类 (5类) ────────────────→ 标准5类
  ├── OIL LEAKAGE ────────────────────────→ erosion
  ├── dirt ───────────────────────────────→ erosion
  ├── Paint ──────────────────────────────→ peeling
  ├── LE-Erosion ─────────────────────────→ erosion
  └── PU-tape ────────────────────────────→ peeling
```

---

## 四、数据集组合后预期

### 类别覆盖分析

| 标准类别 | Blade30 | QQ6 | QQ5 | WTBD | 总计 |
|----------|---------|-----|-----|------|------|
| crack | ✓ | ✓ (1,144) | ✗ | ? | ~1,144+ |
| erosion | ✓ | ✓ (233+693+574) | ✓ (753+846+617) | ? | ~3,716+ |
| lightning | ✓ | ✗ | ✗ | ? | 仅Blade30 |
| peeling | ✓ | ✓ (329) | ✓ (2,455+700) | ? | ~3,484+ |
| hole | ✓ | ✓ (303) | ✗ | ? | ~303+ |

### 预计总量

| 指标 | 预估值 |
|------|--------|
| 总图片数 | ~9,000+ (Blade30 1,302 + QQ6 3,282 + QQ5 4,467) |
| 总标注数 | ~15,000+ |
| 类别数 | 5 |
| 格式 | YOLO txt |

### 类别不平衡问题

| 类别 | 预计数量 | 平衡状态 |
|------|----------|----------|
| erosion | ~3,700+ | ⚠️ 过多 |
| peeling | ~3,400+ | ⚠️ 过多 |
| crack | ~1,100+ | ✓ 合理 |
| hole | ~300+ | ❌ 过少 |
| lightning | ~100+ | ❌ 严重不足 |

**解决方案**:
1. 使用加权损失函数 (EIoU + class weights)
2. 对少数类别进行过采样
3. 使用 CopyPaste 数据增强
4. 寻找更多 lightning 和 hole 类别数据

---

## 五、行动清单

### 优先级 1 (必须完成)

- [ ] 下载 Blade30 数据集 (百度网盘)
  - 链接: `https://pan.baidu.com/s/17kv5Xadz1QcSrvoG58WtBw` (码:1234)
  - 解压到: `data/raw/blade30/`
- [ ] 下载 QQ767172261 六类数据集
  - 联系作者或搜索 AI Studio / 魔搭社区
  - 放到: `data/raw/qq767172261_6cls/`
- [ ] 下载 QQ767172261 UAV五类数据集
  - 联系作者或搜索 AI Studio / 魔搭社区
  - 放到: `data/raw/qq767172261_uav5/`

### 优先级 2 (推荐完成)

- [ ] 获取 WTBD 数据集
  - 访问: https://www.nature.com/articles/s41597-026-06762-x
  - 查找 Data Availability 部分
- [ ] 搜索更多 lightning 类别数据
  - 关键词: "风电叶片 雷击 缺陷 数据集"

### 优先级 3 (可选)

- [ ] 搜索 sbinhigher 实例分割数据集
  - 含 lightning 类，5级风险等级
- [ ] 搜索 Roboflow 上的叶片缺陷数据集

---

## 六、下载完成后的处理流程

```bash
# 1. 进入数据目录
cd E:\yolo-wind-turbine\data

# 2. 运行数据清洗
python scripts/clean_data.py --input raw/blade30 --output cleaned/blade30
python scripts/clean_data.py --input raw/qq767172261_6cls --output cleaned/qq6
python scripts/clean_data.py --input raw/qq767172261_uav5 --output cleaned/qq5

# 3. 格式转换
python scripts/convert_format.py --input raw/blade30 --output processed/detection --format json
python scripts/convert_format.py --input raw/qq767172261_6cls --output processed/detection --format yolo
python scripts/convert_format.py --input raw/qq767172261_uav5 --output processed/detection --format yolo

# 4. 合并数据集
python scripts/merge_datasets.py \
    --input raw/blade30 raw/qq767172261_6cls raw/qq767172261_uav5 \
    --output merged

# 5. 划分数据集
python scripts/split_dataset.py --input merged --output processed/detection

# 6. EDA分析
python scripts/eda_report.py --input processed/detection --output docs
```

---

## 七、技术栈确认

| 组件 | 选择 | 版本 |
|------|------|------|
| 检测模型 | YOLOv11 | Ultralytics |
| 轻量化 | GhostNet | CVPR 2020 |
| 注意力 | CA (Coordinate Attention) | ICCV 2021 |
| 特征融合 | BiFPN | 2020 |
| 损失函数 | EIoU + DFL | - |
| 推理优化 | SAHI | 切片推理 |

---

## 八、参考文献

1. Yang et al., "Blade30: A dataset for wind turbine blade inspection", Renewable Energy, 2023
2. QQ767172261, "风电叶片缺陷检测数据集", GitHub
3. Nature Scientific Data 2026, "WTBD Dataset"
4. Hou et al., "Coordinate Attention", ICCV 2021
5. Han et al., "GhostNet", CVPR 2020
