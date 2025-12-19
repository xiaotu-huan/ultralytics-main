# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# simam.py
import torch
import torch.nn as nn


class SimAM(nn.Module):
    """SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        _b, _c, h, w = x.size()
        w * h - 1

        # Calculate mean and variance
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_var = x.var(dim=[2, 3], keepdim=True)

        # Compute attention weights
        attention = (x - x_mean).pow(2) / (4 * (x_var + self.e_lambda)) + 0.5
        attention = self.activation(attention)

        return x * attention


class Bottleneck_SimAM(nn.Module):
    """Standard Bottleneck with SimAM attention."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # 延迟导入
        from ultralytics.nn.modules import Conv

        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.simam = SimAM()

    def forward(self, x):
        return x + self.simam(self.cv2(self.cv1(x))) if self.add else self.simam(self.cv2(self.cv1(x)))


class C3k_SimAM(nn.Module):
    """C3k Bottleneck with SimAM attention."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        # 延迟导入
        from ultralytics.nn.modules import Bottleneck, Conv

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)
        self.simam = SimAM()

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.simam(self.cv3(torch.cat((y1, y2), 1)))


class C3k2_SimAM(nn.Module):
    """C3k2 module with SimAM in the last bottleneck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.n = n

        # 延迟导入
        from ultralytics.nn.modules import Conv

        # 第一个卷积：将输入通道c1分成两部分，每部分self.c通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # 第二个卷积：最终输出通道为c2
        # 输入通道 = 2 * self.c (初始分割) + n * self.c (每个bottleneck输出)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 创建bottleneck模块列表
        self.m = nn.ModuleList()
        for i in range(n):
            if i == n - 1:  # 最后一个bottleneck使用SimAM
                if c3k:
                    self.m.append(C3k_SimAM(self.c, self.c, 1, shortcut, g, e=1.0, k=3))
                else:
                    self.m.append(Bottleneck_SimAM(self.c, self.c, shortcut, g, k=(3, 3), e=1.0))
            else:
                # 非最后一个使用原始版本
                if c3k:
                    from ultralytics.nn.modules import C3k

                    self.m.append(C3k(self.c, self.c, 1, shortcut, g, e=1.0, k=3))
                else:
                    from ultralytics.nn.modules import Bottleneck

                    self.m.append(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0))

    def forward(self, x):
        # 将输入分成两部分
        y = list(self.cv1(x).split([self.c, self.c], 1))

        # 依次通过每个bottleneck模块
        for m in self.m:
            y.append(m(y[-1]))

        # 拼接所有特征图并输出
        return self.cv2(torch.cat(y, 1))
