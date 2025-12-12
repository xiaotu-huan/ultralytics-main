from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="./pipeline.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolo11_pipe_defect_detection'
)