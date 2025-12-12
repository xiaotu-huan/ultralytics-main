from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="F:/github/ultralytics-main/datasets/visdrone/visdrone.yaml",
    imgsz=1280,  # VisDrone目标较小，建议高分辨率
    batch=8,
    epochs=100,
    # device=0
)