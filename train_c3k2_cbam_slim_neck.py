from ultralytics import YOLO

# 构建模型
model = YOLO("F:/github/ultralytics-main/ultralytics/cfg/models/11/c3k2_cbam_slim_neck.yaml")
model.load('yolo11n.pt')  # 加载预训练权重

model.train(
    data="./pipeline.yaml",
    epochs=200,
    imgsz=640,
    batch=-1,            #自动选择适合的batch 
    name='yolo11_c3k2_cbam_slim_neck',
    
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