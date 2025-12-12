from ultralytics import YOLO

# 构建模型并加载预训练权重
model = YOLO("F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11n_c2f_cbam_pretrain.yaml")  
model.load('yolo11n.pt') 



model.train(
    data="./pipeline.yaml",  
    epochs=150,  
    imgsz=640,   
    batch=32,    
    name='yolo11n_c2f_cbam_pipe_defect_detection'  
)

