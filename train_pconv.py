from ultralytics import YOLO

model = YOLO("F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11-pconv.yaml")
model.load('yolo11n.pt')  # 加载预训练权重



model.train(
    data="./pipeline.yaml",
    epochs=200,
    imgsz=640,
    batch=8,           
    name='yolo11-pconv',

    # 学习率调整
    lr0=0.01,           
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
    
    # 优化器
    optimizer="SGD",
    momentum=0.9,        # 明确指定动量
    weight_decay=0.0005, # 权重衰减
    
    # 训练策略
    patience=20,
)

# test_pconv_fixed.py
# import torch
# from ultralytics.nn.modules.pconv import PConv

# def test_pconv_with_scaling():
#     """测试PConv在YOLO缩放机制下的表现"""
#     print("=== 测试PConv与YOLO缩放兼容性 ===")
    
#     # 模拟YOLO的宽度缩放
#     width_multiple = 0.25
    
#     # 原始配置中的通道数
#     config_channels = 512
#     # 缩放后的实际通道数
#     actual_channels = int(config_channels * width_multiple)
    
#     print(f"配置通道: {config_channels}, 实际通道: {actual_channels}")
    
#     # 创建PConv
#     pconv = PConv(actual_channels, config_channels, 3, 1, 4, 'split_cat')
    
#     # 测试前向传播
#     x = torch.randn(2, actual_channels, 16, 16)
#     y = pconv(x)
    
#     print(f"输入形状: {x.shape}")
#     print(f"输出形状: {y.shape}")
#     print(f"PConv实际配置: c1={pconv.c1}, c2={pconv.c2}")
    
#     assert x.shape == y.shape, "PConv改变了形状!"
#     print("✅ PConv与YOLO缩放兼容性测试通过!")

# if __name__ == "__main__":
#     test_pconv_with_scaling()



# debug_model_sizes.py
# from ultralytics.nn.tasks import DetectionModel
# import torch

# def debug_layer_sizes(cfg_path):
#     """调试每层的输出尺寸"""
#     model = DetectionModel(cfg_path)
#     print("模型构建成功，开始调试尺寸...")
    
#     # 创建测试输入
#     x = torch.randn(1, 3, 640, 640)
    
#     # 逐层前向传播并记录尺寸
#     print("\n=== Backbone 层尺寸 ===")
#     for i, layer in enumerate(model.model[:11]):  # backbone部分
#         x = layer(x)
#         print(f"层 {i}: {x.shape}")
    
#     print("\n=== Head 层尺寸 ===")
#     for i, layer in enumerate(model.model[11:], 11):  # head部分
#         try:
#             if hasattr(layer, 'f'):
#                 print(f"层 {i}: {layer.__class__.__name__}, from: {layer.f}")
#             x = layer(x)
#             if isinstance(x, (list, tuple)):
#                 print(f"层 {i}: 输出多个特征图 {[t.shape for t in x]}")
#             else:
#                 print(f"层 {i}: {x.shape}")
#         except Exception as e:
#             print(f"层 {i} 错误: {e}")
#             break

# if __name__ == "__main__":
#     debug_layer_sizes("ultralytics/cfg/models/11/yolo11-pconv.yaml")

# test_pconv.py
# from ultralytics import YOLO

# def test_pconv_config():
#     try:
#         model = YOLO("ultralytics/cfg/models/11/yolo11-pconv.yaml")
#         print("✅ PConv配置文件加载成功!")
        
#         # 测试模型构建
#         model = model.model
#         print("✅ 模型构建成功!")
        
#         # 打印模型信息
#         print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
#     except Exception as e:
#         print(f"❌ 错误: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     test_pconv_config()