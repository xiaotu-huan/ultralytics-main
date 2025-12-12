# ultralytics/nn/modules/pconv.py
import torch
import torch.nn as nn

__all__ = ['PConv']

class PConv(nn.Module):
    """
    Partial Convolution (PConv) from FasterNet
    Only applies convolution to a subset of input channels
    """
    def __init__(self, c1, c2, k=3, s=1, n_div=4, forward='split_cat', p=None, g=1, bias=False):
        super().__init__()
        # 在YOLO中，c2可能被width_multiple缩放，但PConv需要c1 == c2
        # 所以我们使用c1作为基准，忽略c2（因为PConv输入输出通道必须相同）
        self.c1 = c1
        self.c2 = c1  # 强制输出通道等于输入通道
        self.stride = s
        
        self.dim_conv3 = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv3
        
        # 设置padding，如果未指定则自动计算
        if p is None:
            p = k // 2  # 自动计算padding
        
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, 
            kernel_size=k, stride=s, 
            padding=p, groups=g, bias=bias
        )
        
        # 🔥 关键修复：如果stride>1，需要对未卷积部分也进行下采样
        if s > 1:
            self.downsample_untouched = nn.AvgPool2d(kernel_size=s, stride=s)
        else:
            self.downsample_untouched = nn.Identity()
        
        # 选择前向传播方式
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError(f"Forward type {forward} not implemented")

    def forward_slicing(self, x):
        # 仅用于推理，更高效
        x = x.clone()  # 保持原始输入不变
        
        if self.stride > 1:
            # 对卷积部分进行卷积，对未卷积部分进行下采样
            x_conv = self.partial_conv3(x[:, :self.dim_conv3, :, :])
            x_unconv = self.downsample_untouched(x[:, self.dim_conv3:, :, :])
            x = torch.cat([x_conv, x_unconv], 1)
        else:
            x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        
        return x

    def forward_split_cat(self, x):
        # 用于训练和推理
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x2 = self.downsample_untouched(x2)  # 🔥 对未卷积部分也进行下采样
        x = torch.cat((x1, x2), 1)
        return x

    def extra_repr(self):
        return f'c1={self.c1}, c2={self.c2}, stride={self.stride}, dim_conv3={self.dim_conv3}'