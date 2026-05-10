# YOLO + VisDrone 低空小目标检测

基于 YOLOv8 的低空无人机目标检测，使用 VisDrone2019-DET 数据集。

## 项目结构

```
├── download_data.py          # 数据集下载与格式转换
├── setup_env.bat             # 一键环境配置
├── requirements.txt          # Python 依赖
├── data/visdrone/            # 数据集
│   ├── visdrone.yaml         # YOLO 数据集配置
│   ├── images/               # 图片 (train/val/test)
│   └── labels/               # YOLO 标注
├── src/
│   ├── train.py              # 训练脚本
│   ├── detect.py             # 推理脚本
│   ├── evaluate.py           # 评估脚本
│   ├── visualize.py          # 可视化脚本
│   └── utils.py              # 工具函数
├── configs/                  # 模型配置
├── results/                  # 实验结果
├── docs/                     # 文档
└── ppt/                      # 结题汇报
```

## 快速开始

```bash
# 1. 环境配置
setup_env.bat

# 2. 下载数据集
python download_data.py

# 3. 训练基线模型
python src/train.py --model yolov8n --epochs 100

# 4. 评估
python src/evaluate.py --weights runs/baseline/yolov8n_visdrone/weights/best.pt

# 5. 可视化
python src/visualize.py detect --weights runs/baseline/yolov8n_visdrone/weights/best.pt --source data/visdrone/images/val
```

## 类别

| ID | 类别 | 中文 |
|----|------|------|
| 0 | pedestrian | 行人 |
| 1 | people | 人群 |
| 2 | bicycle | 自行车 |
| 3 | car | 汽车 |
| 4 | van | 面包车 |
| 5 | truck | 卡车 |
| 6 | tricycle | 三轮车 |
| 7 | awning-tricycle | 篷三轮车 |
| 8 | bus | 公交车 |
| 9 | motor | 摩托车 |
