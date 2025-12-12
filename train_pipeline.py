# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="F:/github/ultralytics-main/datasets/visdrone/pipe_defect.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="pipe_defect_detection",
)
