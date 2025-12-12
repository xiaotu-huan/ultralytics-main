from ultralytics import YOLO

model = YOLO("F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11_slim_neck.yaml")
model.load('yolo11n.pt')  # 加载预训练权重


model.train(
    data="./pipeline.yaml",
    epochs=200,
    imgsz=640,
    batch=-1,            #自动选择适合的batch 
    # name='yolo11_slim_neck',
    name='yolo11_slim_neck_focaleiou',


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
    momentum=0.9,        # 明确指定动量
    weight_decay=0.0005, # 权重衰减
    
    # 训练策略
    patience=20,
)


