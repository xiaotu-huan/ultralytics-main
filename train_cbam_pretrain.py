# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

# 构建模型并加载预训练权重
model = YOLO("F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11n_cbam_pretrain.yaml")
model.load("yolo11n.pt")


#  开始训练！
#  现在，主要训练的是未被冻结的层：CBAM模块、Neck、Head等。
model.train(data="./pipeline.yaml", epochs=100, imgsz=640, batch=32, name="yolo11n_cbam_pipe_defect_detection")


# # 检查官方模型
# model_official = YOLO('yolo11n.pt')
# print(model_official.model)


# # 检查您的模型
# model_yours = YOLO("F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11n_cbam.yaml")
# print(model_yours.model)


# from ultralytics import YOLO

# # 加载训练好的模型
# model = YOLO('runs/detect/yolo11n_cbam_pretrain_pipe_defect_detection2/weights/best.pt')

# # 进行验证 - 会自动使用训练时的配置
# results = model.val()
# print(results)
