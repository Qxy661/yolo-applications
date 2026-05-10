"""
CBAM (Convolutional Block Attention Module) 实现
用于改进 YOLOv8 的小目标检测能力
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力: 学习每个通道的重要性权重"""
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
    """空间注意力: 学习每个空间位置的重要性权重"""
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
    """CBAM: 通道注意力 + 空间注意力"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


class CBAMC2f(nn.Module):
    """带 CBAM 的 C2f 模块 (用于改进 YOLOv8 backbone)"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules import C2f
        self.c2f = C2f(c1, c2, n, shortcut, g, e)
        self.cbam = CBAM(c2)

    def forward(self, x):
        return self.cbam(self.c2f(x))
