# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

from PIL import Image


def convert_annotation(img_path, ann_path, output_dir):
    """将VisDrone标注转换为YOLO格式."""
    try:
        img = Image.open(img_path)
        img_w, img_h = img.size
    except Exception as e:
        print(f"无法打开图片 {img_path}: {e}")
        return

    yolo_lines = []
    try:
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 8:
                    continue

                # 解析VisDrone格式: <xmin>,<ymin>,<width>,<height>,<class>,...
                xmin, ymin, w, h = map(float, parts[:4])
                cls = int(parts[4])

                # 过滤无效标注 (class=0表示忽略区域)
                if cls == 0:
                    continue

                # 转换为YOLO格式: <class> <x_center> <y_center> <width> <height>
                x_center = (xmin + w / 2) / img_w
                y_center = (ymin + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                yolo_lines.append(f"{cls - 1} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # 保存YOLO格式标注
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(ann_path))
        with open(output_path, "w") as f:
            f.writelines(yolo_lines)

    except Exception as e:
        print(f"处理标注文件 {ann_path} 失败: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="转换VisDrone标注为YOLO格式")
    parser.add_argument("--img_dir", required=True, help="图片目录")
    parser.add_argument("--ann_dir", required=True, help="原始标注目录")
    parser.add_argument("--output_dir", required=True, help="YOLO格式输出目录")
    args = parser.parse_args()

    for img_name in os.listdir(args.img_dir):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(args.img_dir, img_name)
            ann_name = os.path.splitext(img_name)[0] + ".txt"
            ann_path = os.path.join(args.ann_dir, ann_name)
            if os.path.exists(ann_path):
                convert_annotation(img_path, ann_path, args.output_dir)
    print("转换完成！")
