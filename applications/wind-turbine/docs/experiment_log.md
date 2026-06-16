# 风电场叶片缺陷检测 — 实验记录

## 实验环境

- GPU: NVIDIA RTX 4060 Laptop (8GB)
- CUDA: 13.0, cuDNN: 9.x
- PyTorch: 2.6.0+cu124
- ultralytics: 8.4.51
- Python: 3.10.20

## 实验计划

| 实验 | 目标 | 模型 | 改进 | 预期mAP |
|------|------|------|------|---------|
| E1: 基线 | 建立基准 | YOLOv11n | 无 | ~0.75 |
| E2: C2PSA增强 | 注意力增强 | YOLOv11n | C2PSA repeat 2→4 | >0.75 |
| E3: 轻量化 | 减少计算量 | YOLOv11n | C3k2 depth缩减 | ~0.70 |
| E4: BiFPN | 特征融合 | YOLOv11n | PANet→BiFPN | >0.78 |
| E5: 综合改进 | 全部改进 | YOLOv11n | C2PSA+BiFPN | >0.80 |
| E6: 对比实验 | 模型对比 | 多版本 | v5n/v8n/v11n | - |

## 实验结果

### E1: 基线实验 ✅ 已完成

**配置**:
- 模型: YOLOv11n (ultralytics默认)
- 输入: 640×640
- Epochs: 150
- Batch: 8
- 优化器: auto (SGD), lr0=0.01, lrf=0.001
- cos_lr: True, warmup_epochs: 5
- 数据增强: Mosaic=1.0, MixUp=0.1, CopyPaste=0.1

**结果**:
- mAP@0.5: **0.7555**
- mAP@0.5:0.95: 0.4466
- Precision: 0.7381
- Recall: 0.6944
- 参数量: 2,624,080 (2.62M)
- GFLOPs: 6.6

**分析**: 基线模型在2类缺陷检测上表现良好，mAP@0.5达到75.55%。

---

### E2: C2PSA注意力增强 ✅ 已完成 (YAML训练) + ⏳ 微调训练中

**YAML训练结果** (已放弃):
- 从YAML构建模型，随机初始化 → mAP50仅0.5708
- 原因：无预训练权重，训练从零开始

**微调训练方案** (进行中):
- 从基线best.pt加载 → 增强C2PSA repeat 1→4 → 微调
- 499/499权重完美迁移
- 学习率: lr0=0.001 (低于基线的0.01)
- 训练150 epochs

---

### E3: 轻量化设计 ⏳ 微调训练中

**微调方案**:
- 从基线best.pt加载 → 减少C3k2深度 → 微调
- 冻结backbone，只训练修改部分
- 训练150 epochs

---

### E4: BiFPN特征融合 ⏳ 微调训练中

**微调方案**:
- 从基线best.pt加载 → 冻结backbone (layers 0-10) → 只训练Neck+Head
- 基线backbone特征保持不变，Head从零训练
- 训练150 epochs

---

### E5: 综合改进 ⏳ 微调训练中

**微调方案**:
- 从基线best.pt加载 → C2PSA增强repeat=4 + 冻结backbone → 微调
- 结合注意力增强和冻结backbone策略
- 训练150 epochs

---

## 微调训练方案 (v2)

由于YAML模型从随机初始化训练效果差，改用微调方案：
1. 从基线best.pt加载预训练权重
2. 按需修改网络结构（C2PSA增强/冻结backbone）
3. 用较低学习率(lr0=0.001)微调150 epochs

## 消融实验

| 组件 | E1(Baseline) | E2(CA) | E3(Light) | E4(BiFPN) | E5(Improved) |
|------|-------------|--------|-----------|-----------|-------------|
| C2PSA repeat | 1 | 4 | 1 | 1 | 4 |
| 冻结backbone | - | - | - | ✅ | ✅ |
| 微调lr0 | 0.01 | 0.001 | 0.001 | 0.001 | 0.001 |
