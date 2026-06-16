# 风电叶片缺陷检测数据集汇总

> 更新时间: 2026-05-23
> 目标: 收集高质量公开数据集，满足5类缺陷检测需求

## 一、已确认可下载的数据集

### 1.1 Blade30 数据集 ⭐⭐⭐⭐⭐

| 项目 | 信息 |
|------|------|
| **来源** | Renewable Energy, Vol.203, 2023 |
| **论文** | https://github.com/cong-yang/Blade30 |
| **图片数量** | 1,302 张 |
| **叶片数量** | 30 片 |
| **标注格式** | JSON (LabelMe格式) |
| **类别** | crack, erosion, lightning, peeling, hole (5类) |
| **特点** | 真实无人机巡检数据，学术质量最高 |

**下载方式**:
- 百度网盘 (推荐):
  - 叶片1-15: `https://pan.baidu.com/s/17kv5Xadz1QcSrvoG58WtBw` (码:1234)
  - 叶片16-30: `https://pan.baidu.com/s/1hzcwdc6sBXOeja3nkfartg` (码:1234)
- Google Drive:
  - 叶片1-15: `https://drive.google.com/uc?id=1HbB4t9xV2oCgSSxR9hMEOU6v9qDfetmR`
  - 叶片16-30: `https://drive.google.com/uc?id=1SwRdMzA7zCkNVlHuWvk8uK6eDToM0mUV`
- OneDrive: `https://1drv.ms/u/s!AoXJBmXKVWu5tmtUzCJULhrtYuIP`
- 备用OneDrive (YOLOv8-CBAM项目): `https://1drv.ms/u/c/f179880780519024/EaxA4fYH9yxJitjAVF6L3acB0v7-Xh6dLptHb6KDfXFR1A?e=MLFzcf`

**推荐理由**: 唯一包含lightning类别的数据集，学术权威！

---

### 1.2 QQ767172261 六类数据集 ⭐⭐⭐⭐

| 项目 | 信息 |
|------|------|
| **来源** | GitHub: QQ767172261 |
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

**获取方式**: 需联系作者或搜索 AI Studio / 魔搭社区

---

### 1.3 QQ767172261 UAV五类数据集 ⭐⭐⭐⭐

| 项目 | 信息 |
|------|------|
| **来源** | GitHub: QQ767172261 |
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

**获取方式**: 需联系作者或搜索 AI Studio / 魔搭社区

---

### 1.4 QQ767172261 UAV识别数据集 ⭐⭐⭐⭐

| 项目 | 信息 |
|------|------|
| **来源** | GitHub: QQ767172261/UAV-Wind-Turbine-Blade-Defect-Identification-Image-Dataset |
| **图片数量** | 3,584 张 |
| **标注格式** | 边界框标注 |
| **类别** | 5类 (LE-Erosion, OIL LEAKAGE, PU-tape, Paint, dirt) |

**获取方式**: 需联系作者或搜索 AI Studio / 魔搭社区

---

### 1.5 mxy021120-ops 风扇缺陷数据集 ⭐⭐⭐

| 项目 | 信息 |
|------|------|
| **来源** | GitHub: mxy021120-ops/fans-defect-Dataset |
| **Roboflow** | https://universe.roboflow.com/saodja/fans-detect/dataset/9 |
| **类别** | 3类 (Dirt, Oil LEAKAGE, Pin Hole) |
| **格式** | YOLO格式 (Roboflow导出) |
| **License** | CC BY 4.0 |

**获取方式**: 直接克隆GitHub仓库或从Roboflow下载

---

### 1.6 StevanHRG YOLOv8-CBAM 处理数据 ⭐⭐⭐

| 项目 | 信息 |
|------|------|
| **来源** | GitHub: StevanHRG/YOLOv8-CBAM |
| **基于** | Blade30数据集 |
| **处理** | 256×256裁剪 + 二值分割掩码 |
| **用途** | 小目标缺陷分割 |

**下载方式**:
- OneDrive: `https://1drv.ms/u/c/f179880780519024/EaxA4fYH9yxJitjAVF6L3acB0v7-Xh6dLptHb6KDfXFR1A?e=MLFzcf`
- Google Drive: `https://drive.google.com/file/d/1kk3LfsfYOELNANeJWW_s89YWiOEDmOV0/view?usp=sharing`

---

### 1.7 Kaggle 数据集 ⭐⭐⭐⭐

| 项目 | 信息 |
|------|------|
| **来源** | Kaggle |
| **链接** | https://www.kaggle.com/datasets?search=wind+turbine+blade+defect |

**推荐数据集**:

| 数据集 | 链接 | 图片数 | 类别 |
|--------|------|--------|------|
| Wind Turbine Blade Defect Detection | `https://www.kaggle.com/datasets/rituparnasarmah/wind-turbine-blade-defect-detection` | ~1,444 | blade, crack, erosion, lightning, surface defects |
| Wind Turbine Blade Damage | `https://www.kaggle.com/datasets/anshtanwar/wind-turbine-blade-damage` | ~800+ | crack, erosion, good |
| Wind Turbine Blade Inspection | `https://www.kaggle.com/datasets/parthdande/wind-turbine-blade-inspection` | ~1,000+ | cracks, erosion, contamination |

**下载方式**:
```bash
pip install kaggle
kaggle datasets download -d rituparnasarmah/wind-turbine-blade-defect-detection
```

---

## 二、待获取数据集

### 2.1 WTBD 数据集 (Nature Scientific Data 2026)

| 项目 | 信息 |
|------|------|
| **来源** | Nature Scientific Data 2026 |
| **论文** | https://www.nature.com/articles/s41597-026-06762-x |
| **状态** | ❌ 未获取 |
| **获取方式** | 访问论文Data Availability部分 |

**注意**: Nature论文通常在figshare或zenodo发布数据，可能有百度网盘链接。

### 2.2 zhaowenhai2023 高分辨率数据集 (百度网盘链接可能失效)

| 项目 | 信息 |
|------|------|
| **来源** | GitHub: zhaowenhai2023/Wind-turbine-blade-surface-defect-dataset |
| **图片数量** | 3,800+ 张高分辨率图片 |
| **特点** | 使用StyleGAN3和PBGM方法增强，背景多样 |
| **百度网盘** | 图片: `https://pan.baidu.com/s/1_tJBlhuNE1eDMxTO9uzdWA` (码:1234) |
|  | 标注: `https://pan.baidu.com/s/1Qr0jRVLFUpa13HH0HDsC5Q` (码:1234) |
| **状态** | ⚠️ 百度网盘链接可能已失效，需确认 |

---

## 三、数据集组合统计

### 3.1 可用数据量

| 数据集 | 图片数 | 格式 | 下载方式 | 优先级 |
|--------|--------|------|----------|--------|
| Blade30 | 1,302 | JSON | 百度网盘 ✅ | ★★★★★ |
| QQ767172261 六类 | 3,282 | YOLO txt | 需联系作者 | ★★★★ |
| QQ767172261 UAV五类 | 4,467 | YOLO txt | 需联系作者 | ★★★★ |
| QQ767172261 UAV识别 | 3,584 | 边界框 | 需联系作者 | ★★★★ |
| Kaggle数据集 | ~3,200+ | YOLO/VOC | Kaggle下载 | ★★★★ |
| mxy021120-ops | 待确认 | YOLO txt | GitHub/Roboflow | ★★★ |
| StevanHRG处理数据 | 待确认 | 裁剪+掩码 | OneDrive/Google | ★★★ |
| **总计** | **~16,000+** | | | |

### 3.2 类别覆盖分析

| 标准类别 | Blade30 | QQ6 | QQ5 | QQ识别 | Kaggle | 总计 |
|----------|---------|-----|-----|--------|--------|------|
| crack | ✓ | ✓ (1,144) | ✗ | ✗ | ✓ | ~1,144+ |
| erosion | ✓ | ✓ (1,500) | ✓ (2,216) | ✓ (1,370) | ✓ | ~5,086+ |
| lightning | ✓ | ✗ | ✗ | ✗ | ✓ | 仅Blade30+Kaggle |
| peeling | ✓ | ✓ (329) | ✓ (3,155) | ✓ (2,216) | ✓ | ~5,700+ |
| hole | ✓ | ✓ (303) | ✗ | ✗ | ✓ | ~303+ |

---

## 三、数据集组合统计

### 3.1 可用数据量

| 数据集 | 图片数 | 格式 | 百度网盘 | 优先级 |
|--------|--------|------|----------|--------|
| zhaowenhai2023 | 3,800+ | 待确认 | ✅ 有 | ★★★★★ |
| Blade30 | 1,302 | JSON | ✅ 有 | ★★★★★ |
| QQ767172261 六类 | 3,282 | YOLO txt | ❌ 需联系 | ★★★★ |
| QQ767172261 UAV五类 | 4,467 | YOLO txt | ❌ 需联系 | ★★★★ |
| QQ767172261 UAV识别 | 3,584 | 边界框 | ❌ 需联系 | ★★★★ |
| mxy021120-ops | 待确认 | YOLO txt | ❌ GitHub | ★★★ |
| StevanHRG处理数据 | 待确认 | 裁剪+掩码 | ❌ OneDrive | ★★★ |
| **总计** | **~16,000+** | | | |

### 3.2 类别覆盖分析

| 标准类别 | zhaowenhai | Blade30 | QQ6 | QQ5 | QQ识别 | 总计 |
|----------|------------|---------|-----|-----|--------|------|
| crack | ? | ✓ | ✓ (1,144) | ✗ | ✗ | ~1,144+ |
| erosion | ? | ✓ | ✓ (1,500) | ✓ (2,216) | ✓ (1,370) | ~5,086+ |
| lightning | ? | ✓ | ✗ | ✗ | ✗ | 仅Blade30 |
| peeling | ? | ✓ | ✓ (329) | ✓ (3,155) | ✓ (2,216) | ~5,700+ |
| hole | ? | ✓ | ✓ (303) | ✗ | ✗ | ~303+ |

---

## 四、下载优先级建议

### 优先级 1: 立即下载 (有百度网盘)

1. **Blade30 数据集** (1,302张)
   - 叶片1-15: `https://pan.baidu.com/s/17kv5Xadz1QcSrvoG58WtBw` (码:1234)
   - 叶片16-30: `https://pan.baidu.com/s/1hzcwdc6sBXOeja3nkfartg` (码:1234)

### 优先级 2: Kaggle下载

2. **Kaggle数据集** (~3,200+张)
   - `https://www.kaggle.com/datasets/rituparnasarmah/wind-turbine-blade-defect-detection`
   - `https://www.kaggle.com/datasets/anshtanwar/wind-turbine-blade-damage`
   - `https://www.kaggle.com/datasets/parthdande/wind-turbine-blade-inspection`

### 优先级 3: 搜索获取

3. **QQ767172261 系列数据集** (共~11,333张)
   - 搜索平台: AI Studio, 魔搭社区, 天池
   - 关键词: "风电叶片缺陷 YOLOv8 QQ767172261"

### 优先级 4: 论文数据

4. **WTBD 数据集** (Nature 2026)
   - 访问: https://www.nature.com/articles/s41597-026-06762-x
   - 查找 Data Availability 部分

---

## 五、百度网盘下载清单

### 已确认链接

| 数据集 | 链接 | 码 | 内容 |
|--------|------|----|----|
| Blade30 叶片1-15 | `https://pan.baidu.com/s/17kv5Xadz1QcSrvoG58WtBw` | 1234 | 651张 |
| Blade30 叶片16-30 | `https://pan.baidu.com/s/1hzcwdc6sBXOeja3nkfartg` | 1234 | 651张 |
| zhaowenhai2023 图片 | `https://pan.baidu.com/s/1_tJBlhuNE1eDMxTO9uzdWA` | 1234 | 3,800+张 (可能失效) |
| zhaowenhai2023 标注 | `https://pan.baidu.com/s/1Qr0jRVLFUpa13HH0HDsC5Q` | 1234 | 标注数据 (可能失效) |

### 待搜索链接

| 数据集 | 搜索关键词 |
|--------|-----------|
| QQ767172261 六类 | "风电叶片 六类 缺陷 数据集 百度网盘" |
| QQ767172261 UAV五类 | "UAV 风电叶片 五类 缺陷 YOLO" |
| WTBD | "WTBD wind turbine blade Nature 百度网盘" |
| 其他 | "风电叶片 缺陷检测 数据集 2024 2025" |

---

## 六、下载后处理流程

```bash
# 1. 下载并解压到对应目录
# Blade30 → data/raw/blade30/
# Kaggle → data/raw/kaggle/

# 2. 数据清洗
python scripts/clean_data.py --input raw/blade30 --output cleaned/blade30
python scripts/clean_data.py --input raw/kaggle --output cleaned/kaggle

# 3. 格式转换
python scripts/convert_format.py --input raw/blade30 --output processed/detection --format json
python scripts/convert_format.py --input raw/kaggle --output processed/detection

# 4. 合并数据集
python scripts/merge_datasets.py \
    --input raw/blade30 raw/qq767172261_6cls raw/qq767172261_uav5 raw/kaggle \
    --output merged

# 5. 划分数据集
python scripts/split_dataset.py --input merged --output processed/detection

# 6. EDA分析
python scripts/eda_report.py --input processed/detection --output docs
```

---

## 七、参考资源

### GitHub仓库

1. [cong-yang/Blade30](https://github.com/cong-yang/Blade30) - 1,302张
2. [QQ767172261/Deep-Learning-How-the-YOLOV8-Model-Trains-Wind-Turbine-Blade-Defect-Detection-Datasets](https://github.com/QQ767172261/Deep-Learning-How-the-YOLOV8-Model-Trains-Wind-Turbine-Blade-Defect-Detection-Datasets-Establish-Dee) - 3,282张
3. [QQ767172261/Deep-Learning-YOLOv8-Model-Training-UAV-Aerial-Wind-Turbine-Blade-Defect-Detection-Dataset](https://github.com/QQ767172261/Deep-Learning-YOLOv8-Model-Training-UAV-Aerial-Wind-Turbine-Blade-Defect-Detection-Dataset-Detection) - 4,467张
4. [QQ767172261/UAV-Wind-Turbine-Blade-Defect-Identification-Image-Dataset](https://github.com/QQ767172261/UAV-Wind-Turbine-Blade-Defect-Identification-Image-Dataset-Wind-Turbine-Blade-Common-Damage-Fan-Blad) - 3,584张
5. [StevanHRG/YOLOv8-CBAM](https://github.com/StevanHRG/YOLOv8-CBAM) - Blade30处理数据
6. [zhaowenhai2023/Wind-turbine-blade-surface-defect-dataset](https://github.com/zhaowenhai2023/Wind-turbine-blade-surface-defect-dataset) - 3,800+张 (百度网盘可能失效)

### Kaggle数据集

1. [Wind Turbine Blade Defect Detection](https://www.kaggle.com/datasets/rituparnasarmah/wind-turbine-blade-defect-detection) - ~1,444张
2. [Wind Turbine Blade Damage](https://www.kaggle.com/datasets/anshtanwar/wind-turbine-blade-damage) - ~800+张
3. [Wind Turbine Blade Inspection](https://www.kaggle.com/datasets/parthdande/wind-turbine-blade-inspection) - ~1,000+张

### 学术论文

1. Yang et al., "Blade30", Renewable Energy, 2023
2. Nature Scientific Data 2026, "WTBD Dataset"
3. Hou et al., "Coordinate Attention", ICCV 2021
4. Han et al., "GhostNet", CVPR 2020
