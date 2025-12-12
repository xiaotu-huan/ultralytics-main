# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

# 构建模型并加载预训练权重
model = YOLO("F:/github/ultralytics-main/ultralytics/cfg/models/11/c3k2_eca.yaml")
model.load("yolo11n.pt")


model.train(
    data="./pipeline.yaml",
    epochs=200,
    imgsz=640,
    batch=32,
    lr0=0.001,
    optimizer="SGD",  # 关键：必须明确指定为"SGD"
    momentum=0.9,  # 动量参数（可选，但建议保持）
    weight_decay=0.0005,  # 权重衰减（可选）
    name="yolo11n_c3k2_eca2",
)


# # 先检查配置文件
# import yaml

# try:
#     with open("F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11n_c3k2_cbam.yaml", 'r') as f:
#         config = yaml.safe_load(f)
#     print("配置文件解析成功")
#     print("包含的键:", list(config.keys()))
# except Exception as e:
#     print(f"配置文件解析错误: {e}")
