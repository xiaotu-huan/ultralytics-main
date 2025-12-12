import os
import glob
import numpy as np
from pathlib import Path

def read_yolo_annotations(file_path):
    """
    读取YOLO格式的标注文件
    返回: list of [class_id, x_center, y_center, width, height, confidence(可选)]
    """
    annotations = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # 转换所有值为float
                        annotation = [float(x) for x in parts]
                        annotations.append(annotation)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return annotations

def calculate_iou(box1, box2):
    """
    计算两个YOLO格式框的IoU
    box: [x_center, y_center, width, height]
    """
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
    
    # 转换为中心坐标格式为角点坐标格式
    def box_to_corners(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return x1, y1, x2, y2
    
    box1_corners = box_to_corners(box1)
    box2_corners = box_to_corners(box2)
    
    # 计算交集区域
    x_left = max(box1_corners[0], box2_corners[0])
    y_top = max(box1_corners[1], box2_corners[1])
    x_right = min(box1_corners[2], box2_corners[2])
    y_bottom = min(box1_corners[3], box2_corners[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def generate_audit_lists(true_labels_dir, pred_labels_dir, output_dir, conf_threshold=0.7, iou_threshold=0.5):
    """
    生成三份审核清单
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有预测文件
    pred_files = glob.glob(os.path.join(pred_labels_dir, "*.txt"))
    print(f"找到 {len(pred_files)} 个预测文件")
    
    # 初始化结果列表
    potential_missed = []       # 漏标清单
    hard_examples = []          # 困难样本清单
    potential_wrong_label = []  # 可能标错清单
    
    # 统计信息
    stats = {
        'total_images': 0,
        'processed': 0,
        'missed_found': 0,
        'hard_found': 0,
        'wrong_label_found': 0
    }
    
    for pred_file in pred_files:
        stats['total_images'] += 1
        filename = os.path.basename(pred_file)
        true_file = os.path.join(true_labels_dir, filename)
        
        # 读取预测和真实标注
        pred_annotations = read_yolo_annotations(pred_file)
        true_annotations = read_yolo_annotations(true_file)
        
        # 1. 查找漏标 (高置信度预测但没有匹配的真实标注)
        for pred_ann in pred_annotations:
            if len(pred_ann) >= 5:  # 有置信度信息
                conf = pred_ann[4]
                pred_class = int(pred_ann[0])
                pred_box = pred_ann[1:5]
                
                if conf >= conf_threshold:
                    # 检查是否与任何真实框匹配
                    matched = False
                    for true_ann in true_annotations:
                        if len(true_ann) >= 4:
                            true_box = true_ann[1:5]
                            iou = calculate_iou(pred_box, true_box)
                            if iou >= iou_threshold:
                                matched = True
                                break
                    
                    if not matched:
                        potential_missed.append({
                            'image_name': filename.replace('.txt', '.jpg'),
                            'txt_name': filename,
                            'pred_class': pred_class,
                            'confidence': conf,
                            'pred_box': pred_box
                        })
                        stats['missed_found'] += 1
        
        # 2. 查找困难样本和可能标错的样本
        for true_ann in true_annotations:
            if len(true_ann) >= 4:
                true_class = int(true_ann[0])
                true_box = true_ann[1:5]
                
                # 查找匹配的预测
                best_match = None
                best_iou = 0
                best_conf = 0
                
                for pred_ann in pred_annotations:
                    if len(pred_ann) >= 5:
                        pred_class = int(pred_ann[0])
                        pred_box = pred_ann[1:5]
                        conf = pred_ann[4]
                        iou = calculate_iou(pred_box, true_box)
                        
                        if iou >= iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_match = (pred_class, conf, iou)
                            best_conf = conf
                
                if best_match:
                    pred_class, conf, iou = best_match
                    
                    # 2.1 困难样本: 有匹配但置信度低
                    if conf < conf_threshold:
                        hard_examples.append({
                            'image_name': filename.replace('.txt', '.jpg'),
                            'txt_name': filename,
                            'true_class': true_class,
                            'pred_class': pred_class,
                            'confidence': conf,
                            'iou': iou
                        })
                        stats['hard_found'] += 1
                    
                    # 2.2 可能标错: 有匹配但类别不同且置信度高
                    elif pred_class != true_class and conf >= conf_threshold:
                        potential_wrong_label.append({
                            'image_name': filename.replace('.txt', '.jpg'),
                            'txt_name': filename,
                            'true_class': true_class,
                            'pred_class': pred_class,
                            'confidence': conf,
                            'iou': iou
                        })
                        stats['wrong_label_found'] += 1
                
                # 2.3 完全没有预测到的真实标注 (也是困难样本)
                else:
                    hard_examples.append({
                        'image_name': filename.replace('.txt', '.jpg'),
                        'txt_name': filename,
                        'true_class': true_class,
                        'true_box': true_box,
                        'confidence': 0.0,
                        'iou': 0.0,
                        'note': 'no_prediction'
                    })
                    stats['hard_found'] += 1
        
        stats['processed'] += 1
        if stats['processed'] % 100 == 0:
            print(f"已处理 {stats['processed']}/{stats['total_images']} 张图片")
    
    # 保存结果到文件
    def save_list(data, filename, header):
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write(f"# {header}\n")
            f.write(f"# 共找到 {len(data)} 个样本\n")
            f.write("# image_name, details...\n\n")
            for item in data:
                if 'pred_class' in item:
                    f.write(f"{item['image_name']} - pred_class:{item['pred_class']} conf:{item['confidence']:.3f}\n")
                else:
                    f.write(f"{item['image_name']} - true_class:{item['true_class']} conf:{item['confidence']:.3f}\n")
    
    # 保存三个清单
    save_list(potential_missed, "potential_missed.txt", "漏标清单 - 模型高置信度预测但标注中没有的样本")
    save_list(hard_examples, "hard_examples.txt", "困难样本清单 - 低置信度匹配或完全没预测到的样本")
    save_list(potential_wrong_label, "potential_wrong_label.txt", "可能标错清单 - 高置信度预测但类别与标注不一致")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("审核清单生成完成！")
    print(f"处理图片总数: {stats['total_images']}")
    print(f"疑似漏标样本: {stats['missed_found']}")
    print(f"困难样本: {stats['hard_found']}")
    print(f"可能标错样本: {stats['wrong_label_found']}")
    print(f"结果保存到: {output_dir}")
    print("="*50)
    
    return potential_missed, hard_examples, potential_wrong_label

def main():
    """
    主函数 - 请根据你的实际情况修改这些路径！
    """
    # 配置路径参数
    TRUE_LABELS_DIR = "F:/github/ultralytics-main/datasets/pipeline_defect/labels/val"
    PRED_LABELS_DIR = "runs/detect/review_rupture_deformation/labels"
    OUTPUT_DIR = "runs/audit_results"
    
    print("开始生成审核清单...")
    print(f"真实标注路径: {TRUE_LABELS_DIR}")
    print(f"预测结果路径: {PRED_LABELS_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("-" * 50)
    
    # 运行分析
    potential_missed, hard_examples, potential_wrong_label = generate_audit_lists(
        true_labels_dir=TRUE_LABELS_DIR,
        pred_labels_dir=PRED_LABELS_DIR,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.7,   # 高置信度阈值
        iou_threshold=0.5     # IoU匹配阈值
    )
    
    # 生成汇总报告
    summary_path = os.path.join(OUTPUT_DIR, "audit_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("审核结果汇总报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"分析时间: {np.datetime64('now')}\n")
        f.write(f"真实标注路径: {TRUE_LABELS_DIR}\n")
        f.write(f"预测结果路径: {PRED_LABELS_DIR}\n\n")
        f.write("统计结果:\n")
        f.write(f"- 疑似漏标样本: {len(potential_missed)} 个\n")
        f.write(f"- 困难样本: {len(hard_examples)} 个\n")
        f.write(f"- 可能标错样本: {len(potential_wrong_label)} 个\n\n")
        f.write("下一步建议:\n")
        f.write("1. 优先审核 'potential_missed.txt' 中的样本，检查是否真的漏标\n")
        f.write("2. 检查 'potential_wrong_label.txt' 中的样本，确认标注类别是否正确\n")
        f.write("3. 分析 'hard_examples.txt' 中的困难样本，考虑是否需要增加类似样本\n")
    
    print(f"汇总报告已保存: {summary_path}")

if __name__ == "__main__":
    main()