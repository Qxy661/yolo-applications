# 风电叶片缺陷检测

> 基于 YOLO 系列的风电叶片表面缺陷检测  
> 属于 [YOLO 小目标检测](https://github.com/Qxy661/yolo-visdrone) 项目子应用  
> 最佳模型：YOLOv8n, mAP@0.5 = 82.80%

---

## 项目结构

```
wind-turbine/
├── src/                              # 训练/推理/评估脚本
│   ├── train.py                      # 基线训练
│   ├── train_improved.py             # 改进训练 (CA/ECA注意力)
│   ├── train_experiments.py          # 对比实验 + 消融实验
│   ├── evaluate.py / detect.py       # 评估与推理
│   ├── visualize_results.py          # 结果可视化
│   ├── utils.py                      # 工具函数
│   └── sam/                          # YOLO+SAM 融合
│       ├── run_yolo_sam.py           # SAM 分割流水线
│       └── analyze_results.py        # 分割结果分析
├── configs/                          # 模型配置
│   ├── yolov11n.yaml                 # 5类基线配置
│   ├── yolov11s.yaml                 # 5类+CA注意力
│   └── ...v2 改进版 YAML
├── models/                           # 改进模型 YAML (v2)
│   ├── yolo11n_ca_v2.yaml            # C2PSA注意力增强
│   ├── yolo11n_bifpn_v2.yaml         # BiFPN特征融合
│   ├── yolo11n_light_v2.yaml         # 轻量化版本
│   └── yolo11n_improved_v2.yaml      # 综合改进
├── data/
│   ├── wind_turbine.yaml             # 5类数据集配置
│   ├── wind_turbine_2cls.yaml        # 2类数据集配置
│   ├── processed/                    # 处理后的 YOLO 格式数据
│   │   ├── detection/                # 5类 (1158图)
│   │   └── detection_2cls/           # 2类 (1096图)
│   └── scripts/                      # 数据处理流水线
├── weights/                          # 训练好的模型权重
│   ├── yolov8n.pt                    # ✅ 最佳模型 (82.80%)
│   ├── yolov11n_baseline.pt          # YOLOv11n 基线
│   ├── yolov11n_freeze.pt            # 冻结策略
│   └── sam_vit_b_01ec64.pth          # SAM 权重
├── runs/detect/                      # 训练输出
│   ├── wind_turbine_2cls/            # 基线训练
│   ├── compare_yolov5n/              # YOLOv5n 对比
│   ├── compare_yolov8n/              # YOLOv8n 对比 (最佳)
│   └── ablation_freeze_*/            # 冻结消融
├── docs/                             # 完整文档
│   ├── comprehensive_survey.md       # 16+篇论文调研
│   ├── experiment_log.md             # 实验日志
│   ├── reference_analysis.md         # 论文深度分析
│   ├── sam_yolo_research_report.md   # SAM+YOLO 研究报告
│   ├── TECHNICAL_REPORT.md           # 技术报告
│   └── data_reports/                 # 数据分析报告
├── ppt/                              # 汇报材料
├── experiments/                      # 对比评估脚本
└── requirements.txt
```

---

## 数据集

### 2 类（当前最佳效果）

| 属性 | 值 |
|------|-----|
| 类别 | crack (裂纹), erosion (腐蚀) |
| 规模 | 1,096 图 (train: 764, val: 166, test: 166) |
| 来源 | Blade30(风电叶片) + WT Blade Defect 合并 |
| 平均标注 | 1.67 目标/图 |

### 5 类（待完善）

包含 crack, erosion, lightning, peeling, hole。其中 peeling 和 hole 在当前数据集中无样本，lightning 仅占 5.2%，需要更多数据。

---

## 实验结果

### 模型对比 (2类, val集)

| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params | GFLOPs |
|:----:|:-------:|:------------:|:---------:|:-----:|:------:|:------:|
| YOLOv11n (Baseline) | 78.01% | 47.67% | 83.37% | 67.69% | 2.62M | 6.3 |
| YOLOv5n | 80.78% | 48.87% | 85.81% | 70.00% | 2.50M | 7.1 |
| **YOLOv8n** 🏆 | **82.80%** | **52.18%** | **90.24%** | **72.37%** | 3.01M | 8.1 |
| YOLOv11n+Freeze | 76.44% | 46.29% | 84.88% | 66.67% | 2.62M | 6.3 |

### 成果文件

| 文件 | 说明 |
|---|---|
| [best.pt](results/best.pt) | 最佳模型权重（YOLOv8n，mAP 82.80%）|
| [training_curve.png](results/training_curve.png) | 训练曲线 |
| [confusion_matrix.png](results/confusion_matrix.png) | 混淆矩阵 |

### 冻结消融实验

| 策略 | mAP@0.5 | vs Baseline |
|:----:|:-------:|:-----------:|
| freeze=0 (全参微调) | 80.35% | +2.97% |
| freeze=5 | 65.82% | -15.63% |
| freeze=10 | 74.79% | -4.13% |
| freeze=11 (全部冻结) | 76.44% | -2.02% |

### 关键发现

1. **YOLOv8n > YOLOv11n** — 小数据集上老版本反而更好，C2PSA 注意力优势未体现
2. **冻结策略失效** — 小数据集上冻结任何层都降低精度，全参数微调最优
3. **YOLOFromYAML 训练失效** — 从 YAML 随机初始化训练效果远差于从 COCO 预训练加载（mAP ~0.43-0.57 vs 0.828）

---

## 快速开始

```bash
# 环境配置 (在 yolo-visdrone 根目录)
setup_env.bat

# 评估最佳模型
cd applications/wind-turbine
python src/evaluate.py --weights weights/yolov8n.pt

# 从头训练
python src/train.py --model yolov8n.pt --epochs 150 --batch 8

# YOLO+SAM 融合分割
python src/sam/run_yolo_sam.py
```

---

## 实验路线

```
第1阶段 (已完成): 基线 + 模型对比
  → YOLOv5n/v8n/v11n 对比，确定 YOLOv8n 最优

第2阶段 (已完成): 冻结消融
  → 确定小数据集上全参数微调最优

第3阶段 (未完成): 架构改进
  → C2PSA/BiFPN/轻量化/YOLO+FromYAML 均未显著提升
  → 需更换思路：超大模型尝试 / 数据增广 / 5类完善

第4阶段 (探索中): SAM 分割融合
  → YOLO 检测 → SAM 分割，获取像素级缺陷轮廓
```

---

## 硬件环境

- GPU: NVIDIA NVIDIA GPU (8GB) | CUDA 12.4
- PyTorch 2.6.0 | ultralytics 8.4.48
- Python 3.9+

---

## License

MIT License
