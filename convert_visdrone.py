# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import xml.etree.ElementTree as ET

# VisDrone类别映射（官方顺序）
CLASS_MAP = {
    "pedestrian": 0,
    "people": 1,
    "bicycle": 2,
    "car": 3,
    "van": 4,
    "truck": 5,
    "tricycle": 6,
    "awning-tricycle": 7,
    "bus": 8,
    "motor": 9,
    "others": 10,
}


def convert(xml_path, txt_output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(txt_output_path, "w") as f:
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASS_MAP:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # 计算YOLO格式（归一化中心坐标+宽高）
            img_width = float(root.find("size/width").text)
            img_height = float(root.find("size/height").text)

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{CLASS_MAP[cls]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# 处理训练集
os.makedirs("datasets/visdrone/train/labels", exist_ok=True)
for xml_file in os.listdir("datasets/visdrone/train/labels"):
    if xml_file.endswith(".xml"):
        img_name = os.path.splitext(xml_file)[0]
        convert(f"datasets/visdrone/train/labels/{xml_file}", f"datasets/visdrone/train/labels/{img_name}.txt")

# 处理验证集（同理）
os.makedirs("datasets/visdrone/val/labels", exist_ok=True)
for xml_file in os.listdir("datasets/visdrone/val/labels"):
    if xml_file.endswith(".xml"):
        img_name = os.path.splitext(xml_file)[0]
        convert(f"datasets/visdrone/val/labels/{xml_file}", f"datasets/visdrone/val/labels/{img_name}.txt")
