# 07 · YOLO 应用典型闭环

> 从"拿到一个检测需求"到"上线使用"的完整流程。
> 这套方法可复用到任何领域（钢珠/无人机/工业检测）——是 YOLO 应用的"标准范例"。

## 前置

- 前 6 篇（检测基础 → YOLO原理 → 小目标 → 数据 → 训练评估 → 部署）

## 核心思想：一个可复用的闭环

YOLO 应用不是零散操作，而是一条**标准流水线**。任何领域都走同一套流程：

```
数据准备 → 训练微调 → 评估优化 → 部署上线
```

## 一、数据准备（分析/自制）

### 1. 数据获取
- **公开数据集**：HuggingFace / Roboflow / Kaggle（搜索目标关键词）
- **自制数据**：相机采集 + 录制视频抽帧（效率高）

### 2. 数据清洗（本项目的实践）
以钢珠数据为例，清洗三步：
```
① 格式统一：LabelMe json → YOLO txt（写转换脚本）
② 去重/合并：json 和 txt 标注合并，去重
③ 过滤无效：越界框 / 零面积框 / 无效标注
```
**本项目结果**：430 张 → **1943 张**（json 转换 + 清洗，扩 4.5 倍）。

### 3. 数据划分
```
train / val / test = 8 : 1 : 1
```
- **train**：训练
- **val**：调参验证（不参与训练）
- **test**：最终评估（训练全程不碰）

### 4. 配置 data.yaml
```yaml
path: /path/to/data
train: images/train
val: images/val
nc: 1
names: {0: steel_ball}
```

## 二、训练（选基线 + 微调）

### 1. 选基线模型
```
yolov8n.pt / yolov8s.pt / yolo26n.pt / yolo26s.pt ...
```
**原则**：
- 数据少/要求快 → 小模型（n）
- 数据多/要求精度 → 大模型（s/m/l）
- 用 COCO 预训练权重（不要从零）

### 2. 微调训练
```python
from ultralytics import YOLO
model = YOLO("yolo26s.pt")       # 基线
model.train(data="data.yaml",
            imgsz=640,           # 匹配数据分辨率
            epochs=50,
            batch=8,
            mosaic=1.0,          # 增强
            device=0)            # GPU
```
训练后得到 `best.pt`（最优权重）。

### 3. 调参优化（公认规范）
| 参数 | 规范做法 |
|---|---|
| imgsz | 小目标/大图 → 大值；普通 → 640 |
| 增强 | mosaic + 旋转 + HSV，**适度**（过度反降）|
| patience | 早停，val loss 不降就停 |
| close_mosaic | 最后 N 轮关 mosaic 防过拟合 |

## 三、评估（验证效果）

```python
model.val(data="data.yaml")   # 输出 mAP50 / mAP50-95 / 各类别 AP
```

| 指标 | 看什么 |
|---|---|
| mAP@0.5 | 整体检测效果 |
| mAP@0.5:0.95 | 定位精度（更严）|
| **AP_small** | 小目标效果（小目标任务重点）|

**优化手段**（效果不够时）：
- 阈值搜索（val 上网格找 conf）
- SAHI 切片推理（大图小目标）
- 数据增强调整

## 四、部署（上线使用）

### 1. 导出格式
```python
model.export(format="onnx")       # 通用
model.export(format="engine", half=True)  # TensorRT GPU加速
```

### 2. 实时推理
```python
from ultralytics import YOLO
model = YOLO("best.engine")
results = model.predict(source=0, stream=True, conf=0.25)
for r in results:
    boxes = r.boxes  # 坐标 + 类别
    # 业务逻辑（输出坐标/触发动作）
```

### 3. 与业务集成
- 检测输出坐标 → 控制逻辑
- 检测输出类别 → 分类决策

## 五、本项目（VisDrone）走这条闭环的实践

```
数据：VisDrone（已有）→ 清洗 → train/val
训练：yolo26s.pt 微调 → best.pt（mAP50 0.44+）
评估：mAP50 / mAP50-95 / AP_small
部署：（待做）ONNX/TensorRT + 实时推理
```

## 六、复用到其他领域（范例的价值）

这套闭环是**领域无关**的。换一个任务（钢珠/风电/工业检测）只需：
```
换数据集（获取+清洗）→ 换 data.yaml → 训练微调 → 评估 → 部署
```
**流程完全一样，只换数据**——这就是"典型闭环"作为范例的价值。

## 小结

- YOLO 应用 = 数据准备 → 训练微调 → 评估优化 → 部署上线
- 数据清洗是关键（格式转换/去重/过滤）
- 选基线 + 微调是训练核心
- mAP/AP_small 评估效果
- 部署导出 + 实时推理
- **流程领域无关，可复用到任何检测任务**

## 下一步

回到项目：完成 VisDrone 的闭环（评估 + 部署），再复用到钢珠检测。
