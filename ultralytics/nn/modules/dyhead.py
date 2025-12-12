# dyhead.py
import torch
import torch.nn as nn

class DyHead(nn.Module):
    """简化但有效的DyHead模块"""
    
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        reduced_channels = max(in_channels // reduction, 16)
        
        # 简化版尺度注意力
        self.level_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 简化版空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.enhance_conv = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        # 保留原始特征
        residual = x
        
        # 尺度注意力
        level_attn = self.level_attention(x)
        level_out = x * level_attn
        
        # 空间注意力
        spatial_attn = self.spatial_attention(level_out)
        spatial_out = level_out * spatial_attn
        
        # 特征增强 + 残差连接
        enhanced = self.enhance_conv(spatial_out)
        return enhanced + residual