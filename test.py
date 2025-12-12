from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")  # 或替换为其他模型如 'yolov8s-seg.pt'（分割任务）

# 预测
# 检测图片
results = model("F:/github/ultralytics-main/ultralytics/assets/000001_0.jpg", save=True)
# 检测视频
# results = model(source="./ultralytics/assets/破相还会要我吗.mp4")
# 检测屏幕
# result = model(source="screen")
# 检测摄像头
# results = model(source=0)

# 训练
# model.train(data="coco8.yaml", epochs=100)  

# 导出模型（如ONNX）
# model.export(format="onnx")