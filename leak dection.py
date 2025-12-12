import os
from PIL import Image

def analyze_problem_images():
    problem_images = [
        "001363.jpg", "003742_1.jpg", "004652.jpg", "004992_2.jpg",
        "002892_2.jpg", "004232_2.jpg", "004847_0.jpg"
    ]
    
    image_dir = "F:/github/ultralytics-main/datasets/pipeline_defect/images/val"
    label_dir = "F:/github/ultralytics-main/datasets/pipeline_defect/labels/val"
    
    for img_name in problem_images:
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        
        print(f"\n=== 分析 {img_name} ===")
        
        # 检查图片尺寸和标注框大小
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                width, height = img.size
                print(f"图片尺寸: {width}x{height}")
        
        # 检查标注框信息
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                print(f"标注框数量: {len(lines)}")
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])
                        # 转换为像素坐标
                        bbox_width = w * width
                        bbox_height = h * height
                        print(f"  框{i+1}: 类别{class_id}, 尺寸{bbox_width:.1f}x{bbox_height:.1f}像素")

analyze_problem_images()