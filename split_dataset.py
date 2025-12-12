import os
import shutil
from sklearn.model_selection import train_test_split

# 设置路径
data_dir = "datasets/pipeline_defect"
image_train_dir = os.path.join(data_dir, "images", "train")
label_train_dir = os.path.join(data_dir, "labels", "train")

# 创建验证集和测试集目录
image_val_dir = os.path.join(data_dir, "images", "val")
image_test_dir = os.path.join(data_dir, "images", "test")
label_val_dir = os.path.join(data_dir, "labels", "val")
label_test_dir = os.path.join(data_dir, "labels", "test")

os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(image_test_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)
os.makedirs(label_test_dir, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(image_train_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

print(f"总图像数: {len(image_files)}")

if len(image_files) == 0:
    print("错误：没有找到图像文件！")
    print("请检查路径:", image_train_dir)
    exit()

# 首先分割出测试集（20%）
train_val_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

# 再从剩余数据中分割出验证集（25%的训练验证数据作为验证集，即总数据的20%）
train_files, val_files = train_test_split(train_val_files, test_size=0.25, random_state=42)

print(f"训练集: {len(train_files)} 张图像")
print(f"验证集: {len(val_files)} 张图像") 
print(f"测试集: {len(test_files)} 张图像")

# 移动文件到验证集目录
val_count = 0
for file in val_files:
    # 移动图像文件
    src_image = os.path.join(image_train_dir, file)
    dst_image = os.path.join(image_val_dir, file)
    shutil.move(src_image, dst_image)
    
    # 移动对应的标注文件
    label_file = os.path.splitext(file)[0] + '.txt'
    src_label = os.path.join(label_train_dir, label_file)
    dst_label = os.path.join(label_val_dir, label_file)
    
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
        val_count += 1

# 移动文件到测试集目录
test_count = 0
for file in test_files:
    # 移动图像文件
    src_image = os.path.join(image_train_dir, file)
    dst_image = os.path.join(image_test_dir, file)
    shutil.move(src_image, dst_image)
    
    # 移动对应的标注文件
    label_file = os.path.splitext(file)[0] + '.txt'
    src_label = os.path.join(label_train_dir, label_file)
    dst_label = os.path.join(label_test_dir, label_file)
    
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
        test_count += 1

print(f"\n成功移动:")
print(f"验证集: {val_count} 个图像-标注对")
print(f"测试集: {test_count} 个图像-标注对")
print("数据集划分完成！")
print("划分比例：训练集60%，验证集20%，测试集20%")

# 验证最终的文件数量
print(f"\n最终各集合文件数量:")
print(f"训练集图像: {len(os.listdir(image_train_dir))}")
print(f"验证集图像: {len(os.listdir(image_val_dir))}")
print(f"测试集图像: {len(os.listdir(image_test_dir))}")
