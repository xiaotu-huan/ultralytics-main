# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# ultralytics/nn/modules/slim_neck.py
import torch
import torch.nn as nn

from .conv import Conv
from .ema import EMA  # 导入你的EMA注意力模块


class GSConv(nn.Module):
    """GSConv: 混合标准卷积和深度可分离卷积."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, 1, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    """GS Bottleneck 模块."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数

        # 两个GSConv模块串联
        self.gsconv1 = GSConv(c1, c_, 1, 1)
        self.gsconv2 = GSConv(c_, c2, 3, 1, g)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 两个GSConv串联
        out = self.gsconv2(self.gsconv1(x))

        # 如果满足条件，添加shortcut连接
        return x + out if self.add else out


class GSBottleneck_EMA(nn.Module):
    """GS Bottleneck with EMA attention - 在最后一个GSConv后添加EMA."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数

        # 两个GSConv模块串联
        self.gsconv1 = GSConv(c1, c_, 1, 1)
        self.gsconv2 = GSConv(c_, c2, 3, 1, g)

        # 在第二个GSConv后添加EMA注意力
        self.ema = EMA(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 两个GSConv串联后应用EMA
        out = self.gsconv2(self.gsconv1(x))
        out = self.ema(out)  # 应用EMA注意力

        # 如果满足条件，添加shortcut连接
        return x + out if self.add else out


class VoV_GSCSP(nn.Module):
    """VoV-GSCSP 模块（根据图6结构）."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数

        # 两个1x1卷积分支
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)

        # n个GSBottleneck模块（使用正确的结构）
        self.m = nn.Sequential(*(GSBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        # 最后的1x1卷积
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        # 两个分支
        x1 = self.cv1(x)  # 主分支
        x2 = self.cv2(x)  # shortcut分支

        # 主分支通过n个GSBottleneck
        x1 = self.m(x1)

        # 拼接两个分支
        return self.cv3(torch.cat((x1, x2), 1))


class VoV_GSCSP_EMA(nn.Module):
    """VoV-GSCSP with EMA attention in the last GSBottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数

        # 两个1x1卷积分支
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)

        # 创建n个GSBottleneck，最后一个使用带EMA的版本
        self.m = nn.Sequential()
        for i in range(n):
            if i == n - 1:  # 最后一个瓶颈层使用EMA版本
                self.m.add_module(f"gsb_ema_{i}", GSBottleneck_EMA(c_, c_, shortcut, g, e=1.0))
            else:  # 前n-1个使用普通版本
                self.m.add_module(f"gsb_{i}", GSBottleneck(c_, c_, shortcut, g, e=1.0))

        # 最后的1x1卷积
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        # 两个分支
        x1 = self.cv1(x)  # 主分支
        x2 = self.cv2(x)  # shortcut分支

        # 主分支通过n个GSBottleneck（最后一个带EMA）
        x1 = self.m(x1)

        # 拼接两个分支
        return self.cv3(torch.cat((x1, x2), 1))
