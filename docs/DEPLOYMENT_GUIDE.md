# 部署指南 · YOLO 模型部署原理与实践

> 讲清楚"训练好的模型怎么部署"，从原理到实操。
> 对应应用流程闭环的最后一步：**部署上线**。

## 为什么需要"导出"模型

训练产出 `best.pt`（PyTorch 权重），但**生产环境不能直接用 .pt**，因为：
1. **依赖重**：PyTorch 运行时要装，部署环境（嵌入式/边缘设备）装不了
2. **慢**：PyTorch 推理有框架开销，达不到实时
3. **平台限制**：很多设备没有 Python/PyTorch

所以要把模型**导出**成轻量、跨平台、快跑的格式。

## 部署格式对比

| 格式 | 速度 | 依赖 | 适用 |
|---|---|---|---|
| **.pt** (PyTorch) | 慢 | 需要 PyTorch | 训练/验证 |
| **ONNX** | 快 | 跨平台 | 通用部署 |
| **TensorRT** | 最快 | NVIDIA GPU | 边缘/实时 |

## ONNX 导出

**ONNX（Open Neural Network Exchange）**：模型通用格式，跨框架/平台。

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx")  # 导出 best.onnx
```

导出后模型不依赖 PyTorch，可被 ONNX Runtime / TensorRT 加载。

## TensorRT 加速（可选，NVIDIA 专用）

**TensorRT**：NVIDIA 的推理优化引擎，针对 GPU 极致加速（FP16 量化）。

```python
model.export(format="engine")  # 导出 TensorRT 引擎
```

比 ONNX 更快，但**只支持 NVIDIA GPU**（部署到 Jetson/AGX 等）。

## 推理流程

```
输入图像 → 预处理(letterbox) → 模型推理 → 后处理(NMS) → 输出坐标/类别
```

```python
# 用 ONNX 推理
import onnxruntime as ort
session = ort.InferenceSession("best.onnx")
# 输入 → 输出检测结果

# 用 YOLO 推理（自动处理）
from ultralytics import YOLO
model = YOLO("best.onnx")
results = model("image.jpg")  # 直接出结果
```

## 实时推理（视频/摄像头）

```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model(source=0)  # 0 = 摄像头，实时检测
```

## 部署实操（本仓库）

```bash
# 1. 导出 ONNX
python scripts/deploy_yolo.py export --model best.pt

# 2. 用 ONNX 推理
python scripts/deploy_yolo.py infer --model best.onnx --source image.jpg

# 3. 实时推理（摄像头）
python scripts/deploy_yolo.py infer --model best.pt --source 0
```

## 部署要点

1. **导出不损失精度**：ONNX/TensorRT 保持模型精度，只是换运行方式
2. **速度提升**：TensorRT 可达 PyTorch 的 3-5 倍快
3. **小目标部署**：边缘设备用小模型（yolo26n），换速度
4. **跨平台**：ONNX 可部署到手机/嵌入式/服务器

## 与学习路径衔接

- 部署是闭环最后一环（数据→训练→评估→**部署**）
- 钢珠检测部署：导出 ONNX 供嵌入式使用
- 为 M3 VLA 的"推理"打基础（同样要处理模型部署）
