# 基线训练报告 - YOLOv11n (2类)

> 训练时间: 2026-05-23
> 模型: YOLOv11n (nano)
> 数据集: crack + erosion (1,096张)

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | yolo11n.pt |
| 类别数 | 2 (crack, erosion) |
| 训练集 | 764张 |
| 验证集 | 166张 |
| 测试集 | 166张 |
| Epochs | 100 |
| Batch Size | 8 |
| 图片尺寸 | 640 |
| 早停 | patience=20 |
| GPU | NVIDIA GeForce RTX 4060 Laptop |
| 显存占用 | ~2.1GB |

## 评估结果 (Test Set)

| 指标 | 结果 |
|------|------|
| **mAP@0.5** | **75.55%** |
| mAP@0.5:0.95 | 44.66% |
| Precision | 73.81% |
| Recall | 69.44% |

## 分析

- mAP@0.5 达到 75.55%，超过 70% 基线目标
- Precision (73.81%) > Recall (69.44%)，模型偏向保守预测
- mAP@0.5:0.95 为 44.66%，定位精度有提升空间
- GPU 显存仅用 2.1GB，可尝试增大 batch size

## 改进方向

1. **轻量化**: GhostNet/GSConv 减少参数量
2. **注意力机制**: CA 模块提升小目标检测
3. **特征融合**: BiFPN 提升多尺度检测能力
4. **损失函数**: EIoU/CIoU 提升定位精度

## 模型文件

- 最佳权重: `runs/detect/wind_turbine_2cls/weights/best.pt`
- 最终权重: `runs/detect/wind_turbine_2cls/weights/last.pt`
- 训练日志: `runs/detect/wind_turbine_2cls/`
