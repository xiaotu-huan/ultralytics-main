import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv  # 导入YOLO的Conv类
from .block import Bottleneck, C3k  # 导入官方的Bottleneck和C3k

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class Bottleneck_EMA(Bottleneck):
    """Bottleneck with EMA attention - 替换Bottleneck_CBAM"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        # 完全匹配官方Bottleneck的参数
        super().__init__(c1, c2, shortcut, g, k, e)
        self.ema = EMA(c2)  

    def forward(self, x):
        x1 = super().forward(x)
        return self.ema(x1)  # 在最后一个Bottleneck输出应用EMA

class C3k_EMA(nn.Module):
    """C3k with EMA attention in the last bottleneck - 替换C3k_CBAM"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        
        # 创建前n-1个普通Bottleneck
        modules = []
        for i in range(n - 1):
            modules.append(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0))
        
        # 最后一个使用带EMA的Bottleneck
        if n > 0:
            modules.append(Bottleneck_EMA(c_, c_, shortcut, g, k=(k, k), e=1.0))
        
        self.m = nn.Sequential(*modules)

    def forward(self, x):
        return self.m(x)

class C3k2_EMA(nn.Module):
    """C3k2 module with EMA attention in the last bottleneck - 替换C3k2_CBAM"""
    def __init__(self, c1, c2, n=1, c3k=False, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1) if shortcut else Conv(2 * self.c, c2, 1)
        
        self.m = nn.ModuleList()
        for i in range(n):
            if i == n - 1:  # 最后一个瓶颈层带EMA
                if c3k:
                    # C3k默认使用k=3的卷积核
                    self.m.append(C3k_EMA(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k=3))
                else:
                    self.m.append(Bottleneck_EMA(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0))
            else:  # 前n-1个普通瓶颈层
                if c3k:
                    self.m.append(C3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k=3))
                else:
                    self.m.append(Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))