from ultralytics import YOLO
import torch

# 1. 加载完整的官方预训练模型（结构和权重）
official_model = YOLO('yolo11n.pt')  # 这个对象拥有完整的预训练权重

# 2. 获取它的状态字典（state_dict），这是所有的权重参数
pretrained_sd = official_model.model.state_dict()

# 3. 创建您自己的模型（空结构）
your_model = YOLO('F:/github/ultralytics-main/ultralytics/cfg/models/11/yolo11n_cbam.yaml').model
your_model_sd = your_model.state_dict() # 您的模型随机初始化的权重

# 4. 关键步骤：遍历您自己模型的层
#    如果某层的名称和官方预训练权重的某层名称完全一致，就把官方权重拷贝过来
for your_key in your_model_sd:
    if your_key in pretrained_sd:
        # 只有当权重形状相同时才拷贝
        if your_model_sd[your_key].shape == pretrained_sd[your_key].shape:
            your_model_sd[your_key] = pretrained_sd[your_key]
            print(f"成功加载层: {your_key}")
        else:
            print(f"形状不匹配，跳过: {your_key}")
    else:
        print(f"新增层，随机初始化: {your_key}")

# 5. 将融合好的权重加载回您的模型
your_model.load_state_dict(your_model_sd)
# 6. 保存为您自己的预训练文件！！！
torch.save(your_model_sd, 'yolo11n_cbam_pretrained.pt')

print("新的预训练权重文件 'yolo11n_cbam_pretrained.pt' 已保存！")