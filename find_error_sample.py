# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

# 配置参数
PREDICTIONS_JSON_PATH = "runs/val/error_analysis2/predictions.json"
DATA_YAML_PATH = "pipeline.yaml"
TARGET_CLASS_IDS = [2, 0]  # Rupture=2, Deformation=0
TARGET_CLASS_NAMES = ["Rupture", "Deformation"]
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_DIR = "error_samples"

# 创建输出目录
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# 加载预测结果
with open(PREDICTIONS_JSON_PATH) as f:
    predictions = json.load(f)

# 加载数据集配置
with open(DATA_YAML_PATH) as f:
    data_config = yaml.safe_load(f)

# 获取类别信息
class_names = data_config["names"]
print(f"数据集类别: {class_names}")


# 修复路径获取方式
def get_image_paths(data_config):
    """获取验证集图像路径."""
    val_path = data_config.get("val", "")

    # 如果val是txt文件
    if val_path.endswith(".txt"):
        if os.path.exists(val_path):
            with open(val_path) as f:
                return [line.strip() for line in f.readlines()]
        else:
            print(f"警告: 找不到val文件 {val_path}")
            return []

    # 如果val是目录
    elif os.path.isdir(val_path):
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(Path(val_path).glob(f"*{ext}")))
            image_paths.extend(list(Path(val_path).glob(f"*{ext.upper()}")))
        return [str(path) for path in image_paths]

    else:
        print(f"警告: 无法解析val路径 {val_path}")
        return []


# 获取图像路径
image_paths = get_image_paths(data_config)
print(f"找到 {len(image_paths)} 张验证图像")

# 如果没有找到图像路径，尝试从predictions.json中推断
if not image_paths:
    print("尝试从预测结果中获取图像信息...")
    # 从predictions中提取唯一的image_id
    unique_image_ids = set(pred["image_id"] for pred in predictions)
    image_paths = [f"unknown_image_{img_id}.jpg" for img_id in unique_image_ids]

# 创建图像ID到路径的映射
image_id_to_path = {i: path for i, path in enumerate(image_paths)}


def analyze_error_samples(predictions, target_class_ids):
    """分析错误样本."""
    error_samples = {"low_confidence": [], "missed_detections": [], "potential_errors": []}

    # 按图像ID组织预测结果
    preds_by_image = defaultdict(list)
    for pred in predictions:
        preds_by_image[pred["image_id"]].append(pred)

    # 分析每个图像
    for image_id in range(len(image_paths)):
        preds = preds_by_image.get(image_id, [])

        # 筛选目标类别的预测
        target_preds = [p for p in preds if p["category_id"] in target_class_ids]

        if not target_preds:
            # 没有目标类别预测，可能是漏检
            error_samples["missed_detections"].append(
                {
                    "image_id": image_id,
                    "image_path": image_id_to_path.get(image_id, "unknown"),
                    "reason": "no_target_class_predictions",
                }
            )
        else:
            # 检查每个预测的质量
            for pred in target_preds:
                if pred["score"] < CONFIDENCE_THRESHOLD:
                    error_samples["low_confidence"].append(
                        {
                            "image_id": image_id,
                            "image_path": image_id_to_path.get(image_id, "unknown"),
                            "class_id": pred["category_id"],
                            "class_name": class_names[pred["category_id"]],
                            "confidence": pred["score"],
                            "bbox": pred["bbox"],
                        }
                    )

                # 所有目标类别的预测
                error_samples["potential_errors"].append(
                    {
                        "image_id": image_id,
                        "image_path": image_id_to_path.get(image_id, "unknown"),
                        "class_id": pred["category_id"],
                        "class_name": class_names[pred["category_id"]],
                        "confidence": pred["score"],
                        "bbox": pred["bbox"],
                    }
                )

    return error_samples


def copy_and_visualize_samples(error_samples):
    """复制和可视化样本."""
    # 创建目录
    visualized_dir = Path(OUTPUT_DIR) / "visualized_errors"
    visualized_dir.mkdir(exist_ok=True)

    copied_count = 0

    # 处理potential_errors
    for sample in error_samples["potential_errors"]:
        if "image_path" in sample and sample["image_path"] != "unknown":
            img_path = Path(sample["image_path"])

            # 如果文件存在，尝试复制和可视化
            if img_path.exists():
                # 复制原图
                dst_path = visualized_dir / f"{img_path.stem}_original{img_path.suffix}"
                shutil.copy2(img_path, dst_path)

                # 尝试可视化（如果有bbox）
                if "bbox" in sample:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            bbox = sample["bbox"]
                            x, y, w, h = bbox

                            # 绘制边界框
                            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 3)

                            # 添加标签
                            label = f"{sample['class_name']}: {sample['confidence']:.2f}"
                            cv2.putText(
                                img, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                            )

                            # 保存可视化结果
                            vis_path = visualized_dir / f"{img_path.stem}_{sample['class_name']}_vis.jpg"
                            cv2.imwrite(str(vis_path), img)
                    except Exception as e:
                        print(f"可视化失败 {img_path}: {e}")

                copied_count += 1

    return copied_count


def generate_augmentation_suggestions(error_samples):
    """生成数据增强建议."""
    print("\n=== 数据增强建议 ===")

    low_conf_count = len(error_samples["low_confidence"])
    missed_count = len(error_samples["missed_detections"])

    print(f"低置信度样本: {low_conf_count} 个")
    print(f"可能漏检样本: {missed_count} 个")

    # 分类统计
    rupture_low_conf = len([s for s in error_samples["low_confidence"] if s["class_id"] == 2])
    deformation_low_conf = len([s for s in error_samples["low_confidence"] if s["class_id"] == 0])

    print(f"\nRupture 低置信度: {rupture_low_conf} 个")
    print(f"Deformation 低置信度: {deformation_low_conf} 个")

    print(f"\n针对 {TARGET_CLASS_NAMES} 的增强建议:")
    print("1. 旋转增强 (±45°) - 改善各种角度的识别")
    print("2. 亮度对比度调整 - 适应不同光照条件")
    print("3. 添加噪声和模糊 - 提高鲁棒性")
    print("4. 随机裁剪和缩放 - 增强尺度不变性")
    print("5. 对错误样本进行过采样训练")


# 运行分析
print("开始分析 Rupture 和 Deformation 的错误样本...")
error_results = analyze_error_samples(predictions, TARGET_CLASS_IDS)

# 打印结果摘要
print("\n=== 错误分析结果 ===")
print(f"目标类别: {TARGET_CLASS_NAMES}")
for error_type, samples in error_results.items():
    print(f"{error_type}: {len(samples)} 个样本")

# 复制和可视化样本
print("\n正在处理样本...")
copied_count = copy_and_visualize_samples(error_results)
print(f"成功处理 {copied_count} 个样本")

# 生成数据增强建议
generate_augmentation_suggestions(error_results)

# 保存详细报告
report_path = Path(OUTPUT_DIR) / "error_analysis_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("管道缺陷检测错误分析报告\n")
    f.write("========================\n\n")
    f.write(f"目标类别: {TARGET_CLASS_NAMES}\n")
    f.write(f"验证集图像: {len(image_paths)} 张\n\n")

    f.write("错误统计:\n")
    for error_type, samples in error_results.items():
        f.write(f"{error_type}: {len(samples)} 个样本\n")

    # 按类别详细统计
    f.write("\n按类别详细统计:\n")
    for class_id in TARGET_CLASS_IDS:
        class_name = class_names[class_id]
        class_low_conf = len([s for s in error_results["low_confidence"] if s["class_id"] == class_id])
        f.write(f"{class_name}: {class_low_conf} 个低置信度样本\n")

print("\n分析完成！")
print(f"报告保存到: {report_path}")
print(f"可视化结果保存到: {OUTPUT_DIR}/visualized_errors/")
