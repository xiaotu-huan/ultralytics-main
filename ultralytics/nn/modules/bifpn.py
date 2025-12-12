# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BiFPN_Concat", "Conv2d_BN"]


class Conv2d_BN(nn.Module):
    """Standard Convolution + BatchNorm + SiLU block with AMP support."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        # 确保输入数据类型与权重一致
        if x.dtype != self.conv.weight.dtype:
            x = x.to(self.conv.weight.dtype)

        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

    def forward_fuse(self, x):
        if x.dtype != self.conv.weight.dtype:
            x = x.to(self.conv.weight.dtype)
        return self.act(self.conv(x))


class BiFPN_Concat(nn.Module):
    """修复AMP兼容的BiFPN融合模块（权重强制FP32）."""

    def __init__(self, dimension=1, out_channels=None):
        super().__init__()
        self.dimension = dimension
        self.epsilon = 1e-4
        # 权重强制FP32（AMP下梯度需FP32）
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.act = nn.ReLU()
        self.out_channels = out_channels

        # 卷积层延迟初始化（默认FP32）
        self.conv1 = None
        self.conv2 = None
        self.conv_out = None
        if self.out_channels:
            # 初始化时强制FP32，不依赖输入dtype
            self.conv_out = Conv2d_BN(self.out_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        if not isinstance(x, list) or len(x) != 2:
            if isinstance(x, torch.Tensor):
                return x
            raise ValueError(f"BiFPN_Concat requires exactly 2 inputs, got {len(x)}")

        x1, x2 = x
        device = x1.device
        compute_dtype = x1.dtype  # AMP下是FP16（计算用）
        param_dtype = torch.float32  # 权重强制FP32（保存和梯度用）

        # -------------------------- 1. 动态初始化卷积层（强制FP32）--------------------------
        if self.out_channels:
            # 初始化时权重为FP32，仅在计算时转compute_dtype
            if self.conv1 is None:
                self.conv1 = Conv2d_BN(x1.shape[1], self.out_channels, kernel_size=1).to(device, param_dtype)
            if self.conv2 is None:
                self.conv2 = Conv2d_BN(x2.shape[1], self.out_channels, kernel_size=1).to(device, param_dtype)

        # -------------------------- 2. 尺寸匹配（保持不变）--------------------------
        if x1.shape[2:] != x2.shape[2:]:
            target_size = x2.shape[2:] if x1.shape[2] < x2.shape[2] else x1.shape[2:]
            x1 = F.interpolate(x1, size=target_size, mode="nearest")
            x2 = F.interpolate(x2, size=target_size, mode="nearest")

        # -------------------------- 3. 通道统一（权重转计算dtype）--------------------------
        if self.out_channels:
            # 计算时转FP16，计算后权重仍保持FP32（不修改原权重dtype）
            x1 = self.conv1(x1.to(compute_dtype))
            x2 = self.conv2(x2.to(compute_dtype))
        else:
            min_ch = min(x1.shape[1], x2.shape[1])
            x1 = x1[:, :min_ch, :, :] if x1.shape[1] > min_ch else x1
            x2 = x2[:, :min_ch, :, :] if x2.shape[1] > min_ch else x2

        # -------------------------- 4. 加权融合（权重转计算dtype）--------------------------
        w = self.w.clone().to(device=device, dtype=compute_dtype)  # 权重转FP16计算
        w = self.act(w)
        w = w / (w.sum() + self.epsilon)

        # -------------------------- 5. 输出卷积（保持兼容）--------------------------
        fused = w[0] * x1 + w[1] * x2
        if self.conv_out:
            fused = self.conv_out(fused.to(compute_dtype))

        return fused

    def forward_split(self, x):
        return self.forward(x)
