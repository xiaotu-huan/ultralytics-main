# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


import torch

from ultralytics import YOLO

model = YOLO("F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11-dyhead.yaml")
model.load("yolo11n.pt")  # 加载预训练权重


# 强制初始化DyHeadDetect
def force_init_dyhead_detect(model):
    """强制初始化DyHeadDetect."""
    for m in model.model.modules():
        if hasattr(m, "bias_init"):
            print("=== 强制初始化 DyHeadDetect ===")
            # 设置正确的stride
            m.stride = torch.tensor([8.0, 16.0, 32.0])
            print(f"设置stride: {m.stride}")

            # 强制调用偏置初始化
            m.bias_init()
            print("偏置初始化完成")

            # 检查偏置是否被正确设置
            for i, (a, b, s) in enumerate(zip(m.cv2, m.cv3, m.stride)):
                print(f"层 {i}: 回归偏置范围 [{a[-1].bias.data.min():.3f}, {a[-1].bias.data.max():.3f}]")
                print(f"层 {i}: 分类偏置范围 [{b[-1].bias.data.min():.3f}, {b[-1].bias.data.max():.3f}]")

            # 确保DFL层不被冻结
            if hasattr(m, "dfl") and hasattr(m.dfl, "conv"):
                m.dfl.conv.requires_grad_(True)
                print("DFL层已设置为可训练")


# 应用强制初始化
force_init_dyhead_detect(model)

model.train(
    data="./pipeline.yaml",
    epochs=200,
    imgsz=640,
    batch=8,
    name="yolo11-dyhead",
    # 关键：从头训练或使用预训练backbone
    pretrained=True,  # 使用预训练权重初始化backbone
    # 学习率调整
    lr0=0.01,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
    # 优化器
    optimizer="SGD",
    momentum=0.9,  # 明确指定动量
    weight_decay=0.0005,  # 权重衰减
    # 训练策略
    patience=20,
)


# isolated_test.py
# from ultralytics import YOLO
# import torch

# def isolated_test():
#     print("=== 完全隔离测试 ===")

#     # 只测试我们的模型
#     model_our = YOLO('F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11-dyhead.yaml')
#     our_head = model_our.model.model[-1]

#     print(f"我们的检测头: {type(our_head)}")
#     print(f"参数: {our_head.nc} classes, {our_head.nl} layers")
#     print(f"no: {our_head.no}")

#     # 创建全新的测试输入
#     test_inputs = [
#         torch.randn(1, 64, 80, 80),
#         torch.randn(1, 128, 40, 40),
#         torch.randn(1, 256, 20, 20)
#     ]

#     print("\n输入特征检查:")
#     for i, inp in enumerate(test_inputs):
#         print(f"  输入 {i}: {inp.shape}")

#     # 测试我们的头
#     print("\n=== 测试我们的DyHeadDetect头 ===")
#     our_head.train()
#     with torch.no_grad():
#         try:
#             our_outputs = our_head(test_inputs)
#             print(f"✅ 成功！输出: {len(our_outputs)} layers")
#             for i, out in enumerate(our_outputs):
#                 print(f"  层 {i}: {out.shape}, 范围 [{out.min():.3f}, {out.max():.3f}]")
#         except Exception as e:
#             print(f"❌ 失败: {e}")
#             # 详细调试
#             print("\n=== 详细调试 ===")
#             for i in range(our_head.nl):
#                 print(f"处理层 {i}:")
#                 print(f"  输入形状: {test_inputs[i].shape}")
#                 print(f"  cv2期望输入通道: {our_head.cv2[i][0].conv.in_channels}")
#                 print(f"  cv3期望输入通道: {our_head.cv3[i][0][0].conv.in_channels}")

# if __name__ == '__main__':
#     isolated_test()


# compare_heads.py
# from ultralytics import YOLO
# import torch

# def compare_heads():
#     # 加载官方模型
#     model_official = YOLO('F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml')
#     # 加载我们的模型
#     model_our = YOLO('F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11-dyhead.yaml')

#     # 获取检测头
#     official_head = model_official.model.model[-1]
#     our_head = model_our.model.model[-1]

#     print("=== 官方Detect头 ===")
#     print(f"类型: {type(official_head)}")
#     print(f"参数: {official_head.nc} classes, {official_head.nl} layers")
#     print(f"reg_max: {official_head.reg_max}")
#     print(f"no: {official_head.no}")

#     print("\n=== 我们的DyHeadDetect头 ===")
#     print(f"类型: {type(our_head)}")
#     print(f"参数: {our_head.nc} classes, {our_head.nl} layers")
#     print(f"reg_max: {our_head.reg_max}")
#     print(f"no: {our_head.no}")

#     # 创建测试输入
#     test_inputs = [
#         torch.randn(1, 64, 80, 80),
#         torch.randn(1, 128, 40, 40),
#         torch.randn(1, 256, 20, 20)
#     ]

#     # 测试官方头
#     print("\n=== 测试官方Detect头 ===")
#     official_head.train()
#     with torch.no_grad():
#         official_outputs = official_head(test_inputs)
#         print(f"官方输出: {len(official_outputs)} layers")
#         for i, out in enumerate(official_outputs):
#             print(f"  层 {i}: {out.shape}, 范围 [{out.min():.3f}, {out.max():.3f}]")

#     # 测试我们的头
#     print("\n=== 测试我们的DyHeadDetect头 ===")
#     our_head.train()
#     with torch.no_grad():
#         our_outputs = our_head(test_inputs)
#         print(f"我们的输出: {len(our_outputs)} layers")
#         for i, out in enumerate(our_outputs):
#             print(f"  层 {i}: {out.shape}, 范围 [{out.min():.3f}, {out.max():.3f}]")

# if __name__ == '__main__':
#     compare_heads()


# dyhead测试文件
# from ultralytics.nn.modules import DyHead, DyHeadDetect
# from ultralytics import YOLO

# # 测试 DyHead
# dyhead = DyHead(256)
# print("DyHead 导入成功!")

# # 测试 DyHeadDetect
# detect_head = DyHeadDetect(nc=6, ch=[256, 512, 1024])
# print("DyHeadDetect 导入成功!")


# # 测试配置文件
# model = YOLO('F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11-dyhead.yaml')
# print("配置文件加载成功!")


# check_annotations.py
# import yaml
# import os

# # 检查标注文件
# def check_annotations():
#     with open('./pipeline.yaml', 'r') as f:
#         data_cfg = yaml.safe_load(f)

#     print("数据集配置:", data_cfg)

#     # 检查训练标注文件
#     train_label_path = os.path.join(data_cfg['path'], data_cfg['train'].replace('images', 'labels'))
#     val_label_path = os.path.join(data_cfg['path'], data_cfg['val'].replace('images', 'labels'))

#     print(f"训练标注路径: {train_label_path}")
#     print(f"验证标注路径: {val_label_path}")

#     # 检查第一个标注文件
#     if os.path.exists(train_label_path):
#         label_files = os.listdir(train_label_path)
#         if label_files:
#             first_label = os.path.join(train_label_path, label_files[0])
#             print(f"第一个标注文件: {first_label}")
#             with open(first_label, 'r') as f:
#                 content = f.read().strip()
#                 print(f"标注内容: {content}")

# check_annotations()


# debug_train.py
# from ultralytics import YOLO
# import torch

# def test_forward_pass():
#     # 加载模型但不加载预训练权重
#     model = YOLO('F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11-dyhead.yaml')

#     print("模型结构:")
#     print(f"模型类型: {type(model.model)}")
#     print(f"检测头: {model.model.model[-1]}")
#     print(f"检测头类型: {type(model.model.model[-1])}")

#     # 获取检测头
#     detect_head = model.model.model[-1]

#     # 创建测试输入（模拟P3, P4, P5特征图）
#     test_inputs = [
#         torch.randn(1, 64, 80, 80),   # P3
#         torch.randn(1, 128, 40, 40),  # P4
#         torch.randn(1, 256, 20, 20)   # P5
#     ]

#     print("\n测试输入:")
#     for i, inp in enumerate(test_inputs):
#         print(f"输入 {i}: {inp.shape}")

#     print("\n测试前向传播...")

#     # 测试训练模式
#     detect_head.train()
#     with torch.no_grad():
#         train_outputs = detect_head(test_inputs)
#         print(f"训练模式输出类型: {type(train_outputs)}")
#         if isinstance(train_outputs, (list, tuple)):
#             print(f"训练输出长度: {len(train_outputs)}")
#             for i, out in enumerate(train_outputs):
#                 print(f"训练输出 {i} 形状: {out.shape}")
#         else:
#             print(f"训练输出形状: {train_outputs.shape}")

#     # 测试推理模式
#     detect_head.eval()
#     with torch.no_grad():
#         eval_outputs = detect_head(test_inputs)
#         print(f"\n推理模式输出类型: {type(eval_outputs)}")
#         if isinstance(eval_outputs, (list, tuple)) and len(eval_outputs) == 2:
#             print(f"推理输出: {eval_outputs[0].shape}")
#             print(f"原始特征: {len(eval_outputs[1])} layers")
#             for i, feat in enumerate(eval_outputs[1]):
#                 print(f"特征 {i}: {feat.shape}")
#         else:
#             print(f"推理输出形状: {eval_outputs.shape}")

# if __name__ == '__main__':
#     test_forward_pass()
