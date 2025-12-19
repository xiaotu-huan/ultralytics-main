# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

# 需要优先检查的图片
urgent_images = [
    "001225_0.jpg",
    "005220_1.jpg",
    "000315_1.jpg",
    "001192_2.jpg",  # 高置信度错误
    "001363.jpg",
    "003742_1.jpg",
    "004652.jpg",
    "004992_2.jpg",  # 完全漏检
]

image_dir = "F:/github/ultralytics-main/datasets/pipeline_defect/images/val"

for img in urgent_images:
    img_path = os.path.join(image_dir, img)
    if os.path.exists(img_path):
        print(f"请检查: {img_path}")
    else:
        print(f"找不到: {img}")
