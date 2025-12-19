# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import *
    >>> import torch
    >>> import os
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
"""

# 新增：导入注意力模块
# from .attention_pretrain import CBAM


# # 添加导入
# from .attention_c2f_cbam import CBAM, ChannelAttention, SpatialAttention

# from .cbam import CBAM, C3k2_CBAM, Bottleneck_CBAM # 添加这行,cbam在每一个bottleneck中

from .asff import Detect_ASFF  # 添加ASFF检测头模块相关
from .bifpn import BiFPN_Concat, Conv2d_BN  # 添加bifpn相关
from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .cbam import (  # cbam只放在最后一个瓶颈块中，但是要先判断是bottleneck瓶颈块，还是C3K瓶颈块
    CBAM,
    Bottleneck_CBAM,
    C3k2_CBAM,
    C3k_CBAM,
)
from .conv import (
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)

# 添加DyHeadDetect检测头模块相关
from .dyhead import DyHead

# from .cbam import CBAM, C3k2_CBAM # 添加这行，cbam在c3k2输出端
from .eca import ECA, Bottleneck_ECA, C3k2_ECA, C3k_ECA
from .ema import EMA, Bottleneck_EMA, C3k2_EMA
from .ghostconv import C3k2_GhostConv
from .head import (
    OBB,
    Classify,
    Detect,
    DyHeadDetect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from .odconv import C3k2_ODConv

# 添加PConv模块相关
from .pconv import PConv
from .simam import Bottleneck_SimAM, C3k2_SimAM, C3k_SimAM, SimAM  # 添加SimAM注意力机制相关
from .slim_neck import GSBottleneck, GSBottleneck_EMA, GSConv, VoV_GSCSP, VoV_GSCSP_EMA  # 添加slim_neck相关
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "AIFI",
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CBAM",
    # 添加CBAM相关模块
    "CBAM",
    "CIB",
    "DFL",
    # 添加ECA相关
    "ECA",
    "ELAN1",
    "EMA",
    "MLP",
    "OBB",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "A2C2f",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    # bifpn特征融合模块相关
    "BiFPN_Concat",
    "Bottleneck",
    "BottleneckCSP",
    "Bottleneck_CBAM",
    "Bottleneck_ECA",
    "Bottleneck_EMA",
    "Bottleneck_SimAM",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    # "C2f_CBAM"
    # 添加C3K2_CBAM相关模块
    # 'CBAM',
    "C3k2_CBAM",
    "C3k2_ECA",
    "C3k2_EMA",
    "C3k2_GhostConv",
    "C3k2_ODConv",
    "C3k2_SimAM",
    "C3k_CBAM",
    "C3k_ECA",
    "C3k_EMA",
    "C3k_SimAM",
    "C3x",
    "CBAMBottleneck",
    "CBFuse",
    "CBLinear",
    "ChannelAttention",
    "ChannelAttention",
    "Classify",
    "Concat",
    "ContrastiveHead",
    "Conv",
    "Conv2",
    "Conv2d_BN",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "Detect",
    # 添加ASFF检测头模块相关
    "Detect_ASFF",
    # 添加DyHeadDetect检测头模块相关
    "DyHead",
    "DyHeadDetect",
    "Focus",
    "GSBottleneck",
    "GSBottleneck_EMA",
    # 添加slim-neck相关模块
    "GSConv",
    "GhostBottleneck",
    "GhostConv",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Index",
    "LRPCHead",
    "LayerNorm2d",
    "LightConv",
    "MLPBlock",
    "MSDeformAttn",
    "MaxSigmoidAttnBlock",
    "PConv",
    "Pose",
    "Proto",
    "RTDETRDecoder",
    "RepC3",
    "RepConv",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "Segment",
    # 添加SimAM注意力机制相关
    "SimAM",
    "SpatialAttention",
    "SpatialAttention",
    "TorchVision",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "VoV_GSCSP",
    "VoV_GSCSP_EMA",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
)
