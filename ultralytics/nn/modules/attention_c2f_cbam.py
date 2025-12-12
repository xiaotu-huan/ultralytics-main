import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CBAM', 'ChannelAttention', 'SpatialAttention']

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 确保reduction_ratio不会导致通道数为0
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # 合并并应用sigmoid
        out = self.sigmoid(avg_out + max_out)
        return out.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接并卷积
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先通道注意力，再空间注意力
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out