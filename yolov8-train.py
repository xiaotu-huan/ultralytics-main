# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="yolo-one.yaml", workers=0, epochs=50, batch=16)
