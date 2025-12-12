# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

model = YOLO("yolo11n.pt")


model.train(
    data="./pipeline.yaml",
    epochs=200,
    imgsz=640,
    batch=8,
    name="yolo11n_focaleiouloss",
    # 设置随机种子
    seed=42,  # 可以是任意整数，常用 0, 42, 1234
    deterministic=True,  # 确保可复现性
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
