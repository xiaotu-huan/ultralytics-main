# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

# 需要检查的图片列表
urgent_images = [
    "001225_0.jpg",
    "005220_1.jpg",
    "000315_1.jpg",
    "001192_2.jpg",
    "001363.jpg",
    "003742_1.jpg",
    "004652.jpg",
    "004992_2.jpg",
]

label_dir = "F:/github/ultralytics-main/datasets/pipeline_defect/labels/val"

print("优先级图片的标注信息:")
print("=" * 50)

for img in urgent_images:
    label_file = img.replace(".jpg", ".txt")
    label_path = os.path.join(label_dir, label_file)

    if os.path.exists(label_path):
        with open(label_path, encoding="utf-8") as f:
            lines = f.readlines()
            print(f"\n{img}:")
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    print(f"  类别 {class_id}: {parts[1:5]}")
    else:
        print(f"\n{img}: 无标注文件")
