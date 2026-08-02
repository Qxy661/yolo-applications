"""
注册自定义模块到 Ultralytics，使其可在 YAML 中使用
在加载模型前导入此模块即可:

    import register_custom_modules  # noqa: F401
    model = YOLO("configs/yolov8s-p2.yaml")
"""
import torch
import torch.nn as nn
from pathlib import Path


# ── CBAM 实现 ──────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module (通道+空间注意力)"""
    def __init__(self, c1=0, reduction=16, kernel_size=7):
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        self._built = False
        self.channel_att = None
        self.spatial_att = None

    def _build(self, channels):
        self.channel_att = ChannelAttention(channels, self.reduction).to(self._device)
        self.spatial_att = SpatialAttention(self.kernel_size).to(self._device)
        self._built = True

    def forward(self, x):
        if not self._built:
            self._device = x.device
            self._build(x.shape[1])
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


# ── 注册到 ultralytics ─────────────────────────────────

def load_compatible_weights(model, pretrained_path):
    """
    按形状匹配加载预训练权重，跳过不匹配的层。
    用于 P2+CBAM 等自定义架构加载标准 YOLOv8s 权重。
    """
    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if "model" in pretrained:
        pretrained = pretrained["model"]

    model_state = model.state_dict()
    pretrained_state = pretrained.state_dict() if hasattr(pretrained, "state_dict") else pretrained

    compatible = {}
    skipped = []
    for k, v in pretrained_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            compatible[k] = v
        else:
            skipped.append(k)

    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)

    print(f"[load_compatible_weights] 已加载 {len(compatible)}/{len(model_state)} 个权重")
    if skipped:
        print(f"[load_compatible_weights] 跳过 {len(skipped)} 个不匹配的权重")
    return compatible


def _register():
    """将自定义模块注入 ultralytics 的模块命名空间和解析器"""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    # ultralytics parse_model 用 globals()[m] 查找模块
    # 需要注入到 ultralytics.nn.tasks 的全局命名空间
    tasks.CBAM = CBAM
    modules.CBAM = CBAM

    print("[register_custom_modules] CBAM 已注册")


_register()
