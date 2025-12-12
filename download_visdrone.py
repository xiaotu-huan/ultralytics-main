from roboflow import Roboflow

# 替换为你的API Key（从 Roboflow 账户获取）
rf = Roboflow(api_key="YOUR_API_KEY")  
project = rf.workspace("visdrone").project("visdrone-det")
dataset = project.version(3).download("yolov8")