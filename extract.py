# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import re


def extract_target_classes_from_list(problem_list_file, output_file, target_classes):
    """从问题清单中提取包含目标类别的行."""
    target_images = set()

    with open(problem_list_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if line.startswith("#") or not line:
                continue

            # 检查是否包含目标类别
            for cls in target_classes:
                if f"class:{cls}" in line or f"true_class:{cls}" in line or f"pred_class:{cls}" in line:
                    # 提取图片文件名（第一个字段）
                    image_name = line.split(" - ")[0].strip()
                    target_images.add(image_name)
                    break

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 从 {problem_list_file} 中提取的包含类别 {target_classes} 的图片\n")
        for image in sorted(target_images):
            f.write(f"{image}\n")

    return len(target_images)


def extract_from_annotation_files(true_labels_dir, target_classes, output_file):
    """直接从标注文件中提取包含目标类别的图片."""
    target_images = set()

    # 遍历所有标注文件
    for filename in os.listdir(true_labels_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(true_labels_dir, filename)
            try:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
                    # 检查是否包含目标类别（类别ID开头）
                    for cls in target_classes:
                        if re.search(rf"^{cls}\s", content):
                            image_name = filename.replace(".txt", ".jpg")
                            target_images.add(image_name)
                            break
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 直接从标注文件中提取的包含类别 {target_classes} 的图片\n")
        for image in sorted(target_images):
            f.write(f"{image}\n")

    return len(target_images)


def main():
    # 配置参数
    AUDIT_DIR = "runs/audit_results"
    TRUE_LABELS_DIR = "F:/github/ultralytics-main/datasets/pipeline_defect/labels/val"
    OUTPUT_DIR = "runs/target_review"

    # 目标类别ID - 请根据你的data.yaml确认！
    TARGET_CLASSES = [2, 0]  # 假设2=Rupture, 0=Deformation

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("开始提取目标类别图片...")
    print(f"目标类别: {TARGET_CLASSES} (请确认这是Rupture和Deformation的ID)")
    print("-" * 50)

    # 1. 从三份问题清单中提取目标类别
    problem_files = ["potential_missed.txt", "hard_examples.txt", "potential_wrong_label.txt"]

    all_target_images = set()

    for problem_file in problem_files:
        input_path = os.path.join(AUDIT_DIR, problem_file)
        output_path = os.path.join(OUTPUT_DIR, f"target_{problem_file}")

        if os.path.exists(input_path):
            count = extract_target_classes_from_list(input_path, output_path, TARGET_CLASSES)
            print(f"从 {problem_file} 中提取到 {count} 张目标图片")

            # 收集所有图片名
            with open(output_path, encoding="utf-8") as f:
                for line in f:
                    if not line.startswith("#"):
                        all_target_images.add(line.strip())
        else:
            print(f"警告: 文件 {input_path} 不存在")

    # 2. 直接从标注文件中提取所有包含目标类别的图片
    all_annotation_images = set()
    annotation_output = os.path.join(OUTPUT_DIR, "all_target_from_annotations.txt")
    count = extract_from_annotation_files(TRUE_LABELS_DIR, TARGET_CLASSES, annotation_output)
    print(f"从标注文件中提取到 {count} 张包含目标类别的图片")

    # 读取标注文件中的图片
    with open(annotation_output, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                all_annotation_images.add(line.strip())

    # 3. 合并所有需要审核的图片（问题图片 + 所有目标类别图片）
    final_review_list = sorted(all_target_images.union(all_annotation_images))

    # 保存最终审核清单
    final_output = os.path.join(OUTPUT_DIR, "final_review_list.txt")
    with open(final_output, "w", encoding="utf-8") as f:
        f.write("# 最终审核清单 - 需要人工检查的图片\n")
        f.write(f"# 包含: 1) 三份问题清单中的目标类别图片 2) 所有标注中包含类别 {TARGET_CLASSES} 的图片\n")
        f.write(f"# 总共 {len(final_review_list)} 张图片\n\n")
        for image in final_review_list:
            f.write(f"{image}\n")

    print("-" * 50)
    print("提取完成！")
    print(f"最终需要审核的图片数量: {len(final_review_list)}")
    print(f"结果保存在: {final_output}")
    print("\n下一步建议:")
    print("1. 使用CVAT或其他标注工具加载这些图片进行审核")
    print("2. 重点关注 potential_missed.txt 中的疑似漏标")
    print("3. 检查 potential_wrong_label.txt 中的可能标错样本")


if __name__ == "__main__":
    main()
