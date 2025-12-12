import math
import torch
import torch.nn as nn


from .conv import Conv # 导入YOLO的Conv类
from .block import Bottleneck,C3k # 从block.py中导入标准Bottleneck,C3k

class ECA(nn.Module):
    """高效的通道注意力机制"""
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        self.channel = channel
        
        # 自适应计算卷积核大小
        k_size = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1  # 确保为奇数
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                             padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 形状: [batch, channel, height, width]
        b, c, h, w = x.size()
        
        # 全局平均池化
        y = self.avg_pool(x)  # [b, c, 1, 1]
        
        # 调整形状并应用一维卷积
        y = y.squeeze(-1).transpose(-1, -2)  # [b, 1, c]
        y = self.conv(y)  # 一维卷积处理通道关系
        y = y.transpose(-1, -2).unsqueeze(-1)  # [b, c, 1, 1]
        
        # 生成注意力权重
        y = self.sigmoid(y)
        
        # 应用注意力
        return x * y.expand_as(x)


# 把ECA放在C3k2输出端
# class C3k2_ECA(nn.Module):
#     """C3k2 module with ECA attention at the output"""
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
        
#         # 标准C3k2组件
#         self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(2 * c_, c2, 1, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU()
        
#         # 使用标准Bottleneck（不是Bottleneck_ECA）
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        
#         # 在输出端添加ECA
#         self.eca = ECA(c2)
        
#     def forward(self, x):
#         # 标准C3k2前向传播
#         y1 = self.m(self.cv1(x))
#         y2 = self.cv2(x)
#         output = self.act(self.bn(self.cv3(torch.cat((y1, y2), 1))))
        
#         # 应用ECA注意力
#         output = self.eca(output)
        
#         return output

# # eca只放在最后一个瓶颈块中，但是要先判断是bottleneck瓶颈块，还是C3K瓶颈块
class Bottleneck_ECA(Bottleneck):
    """Bottleneck with ECA attention"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.eca = ECA(c2)

    def forward(self, x):
        x1 = super().forward(x)
        return self.eca(x1)

class C3k_ECA(nn.Module):
    """C3k with ECA attention in the last bottleneck"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        
        # 创建前n-1个普通Bottleneck
        modules = []
        for i in range(n - 1):
            modules.append(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0))
        
        # 最后一个使用带ECA的Bottleneck
        if n > 0:
            modules.append(Bottleneck_ECA(c_, c_, shortcut, g, k=(k, k), e=1.0))
        
        self.m = nn.Sequential(*modules)

    def forward(self, x):
        return self.m(x)

class C3k2_ECA(nn.Module):
    """C3k2 module with ECA attention in the last bottleneck"""
    def __init__(self, c1, c2, n=1, c3k=False, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1) if shortcut else Conv(2 * self.c, c2, 1)
        
        self.m = nn.ModuleList()
        for i in range(n):
            if i == n - 1:  # 最后一个瓶颈层带ECA
                if c3k:
                    # C3k默认使用k=3的卷积核
                    self.m.append(C3k_ECA(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k=3))
                else:
                    self.m.append(Bottleneck_ECA(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0))
            else:  # 前n-1个普通瓶颈层
                if c3k:
                    self.m.append(C3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k=3))
                else:
                    self.m.append(Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

