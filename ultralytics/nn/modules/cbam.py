# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn

from .block import Bottleneck, C3k  # 导入官方的Bottleneck和C3k
from .conv import Conv  # 导入YOLO的Conv类


class ChannelAttention(nn.Module):
    """Channel Attention Module."""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """CBAM Attention Module."""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# cbam放在每个bottleneck中
# class Bottleneck_CBAM(nn.Module):
#     """Bottleneck with CBAM attention"""
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(c_)
#         self.cv2 = nn.Conv2d(c_, c2, 3, 1, 1, groups=g, bias=False)
#         self.bn2 = nn.BatchNorm2d(c2)
#         self.add = shortcut and c1 == c2
#         self.cbam = CBAM(c2)  # 在Bottleneck内集成CBAM
#         self.act = nn.SiLU()

#     def forward(self, x):
#         x1 = self.act(self.bn1(self.cv1(x)))
#         x1 = self.act(self.bn2(self.cv2(x1)))
#         x1 = self.cbam(x1)  # 应用CBAM注意力
#         return x + x1 if self.add else x1

# class C3k2_CBAM(nn.Module):
#     """C3k2 module with CBAM attention in bottlenecks"""
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(2 * c_, c2, 1, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU()
#         self.m = nn.Sequential(*(Bottleneck_CBAM(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

#     def forward(self, x):
#         # print(f"C3k2_CBAM输入形状: {x.shape}")
#         y1 = self.m(self.cv1(x))
#         y2 = self.cv2(x)
#         # print(f"y1形状: {y1.shape}, y2形状: {y2.shape}")
#         output = self.act(self.bn(self.cv3(torch.cat((y1, y2), 1))))
#         # print(f"C3k2_CBAM输出形状: {output.shape}")
#         return output


# cbam放在C3k2输出端
# class C3k2_CBAM(nn.Module):
#     """C3k2 module with CBAM attention at the output"""
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)

#         # 标准C3k2组件
#         self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(2 * c_, c2, 1, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU()

#         # 使用标准Bottleneck，不是Bottleneck_CBAM
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

#         # 在输出端添加CBAM
#         self.cbam = CBAM(c2)

#     def forward(self, x):
#         # 标准C3k2前向传播
#         y1 = self.m(self.cv1(x))
#         y2 = self.cv2(x)
#         output = self.act(self.bn(self.cv3(torch.cat((y1, y2), 1))))

#         # 应用CBAM注意力
#         output = self.cbam(output)

#         return output


# cbam只放在最后一个bottleneck中
# class Bottleneck_CBAM(nn.Module):
#     """Bottleneck with CBAM attention (for the last bottleneck only)"""
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(c_)
#         self.cv2 = nn.Conv2d(c_, c2, 3, 1, 1, groups=g, bias=False)
#         self.bn2 = nn.BatchNorm2d(c2)
#         self.add = shortcut and c1 == c2
#         self.cbam = CBAM(c2)  # CBAM只在最后一个Bottleneck中
#         self.act = nn.SiLU()

#     def forward(self, x):
#         x1 = self.act(self.bn1(self.cv1(x)))
#         x1 = self.act(self.bn2(self.cv2(x1)))
#         x1 = self.cbam(x1)  # 应用CBAM注意力
#         return x + x1 if self.add else x1

# class C3k2_CBAM(nn.Module):
#     """C3k2 module with CBAM attention only in the last bottleneck"""
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(2 * c_, c2, 1, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU()

#         # 创建n-1个普通Bottleneck和1个带CBAM的Bottleneck
#         self.m = nn.Sequential()
#         if n > 0:
#             # 前n-1个使用普通Bottleneck（从block.py导入的）
#             for i in range(n - 1):
#                 self.m.add_module(f'bottleneck_{i}', Bottleneck(c_, c_, shortcut, g, e=1.0))
#             # 最后一个使用带CBAM的Bottleneck
#             self.m.add_module(f'bottleneck_{n-1}', Bottleneck_CBAM(c_, c_, shortcut, g, e=1.0))

#     def forward(self, x):
#         # print(f"C3k2_CBAM输入形状: {x.shape}")
#         y1 = self.m(self.cv1(x))
#         y2 = self.cv2(x)
#         # print(f"y1形状: {y1.shape}, y2形状: {y2.shape}")
#         output = self.act(self.bn(self.cv3(torch.cat((y1, y2), 1))))
#         # print(f"C3k2_CBAM输出形状: {output.shape}")
#         return output


# cbam只放在最后一个瓶颈块中，但是要先判断是bottleneck瓶颈块，还是C3K瓶颈块
class Bottleneck_CBAM(Bottleneck):
    """Bottleneck with CBAM attention."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        # 完全匹配官方Bottleneck的参数
        super().__init__(c1, c2, shortcut, g, k, e)
        self.cbam = CBAM(c2)

    def forward(self, x):
        x1 = super().forward(x)
        return self.cbam(x1)


class C3k_CBAM(nn.Module):
    """C3k with CBAM attention in the last bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)

        # 创建前n-1个普通Bottleneck
        modules = []
        for i in range(n - 1):
            modules.append(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0))

        # 最后一个使用带CBAM的Bottleneck
        if n > 0:
            modules.append(Bottleneck_CBAM(c_, c_, shortcut, g, k=(k, k), e=1.0))

        self.m = nn.Sequential(*modules)

    def forward(self, x):
        return self.m(x)


class C3k2_CBAM(nn.Module):
    """C3k2 module with CBAM attention in the last bottleneck."""

    def __init__(self, c1, c2, n=1, c3k=False, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 现在Conv已经被正确导入
        self.cv2 = Conv((2 + n) * self.c, c2, 1) if shortcut else Conv(2 * self.c, c2, 1)

        self.m = nn.ModuleList()
        for i in range(n):
            if i == n - 1:  # 最后一个瓶颈层带CBAM
                if c3k:
                    # C3k默认使用k=3的卷积核
                    self.m.append(C3k_CBAM(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k=3))
                else:
                    self.m.append(Bottleneck_CBAM(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0))
            else:  # 前n-1个普通瓶颈层
                if c3k:
                    self.m.append(C3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k=3))
                else:
                    self.m.append(Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
