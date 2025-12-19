# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import random

# 读取文件（自动处理编码）
encodings = ["utf-8", "gbk", "latin-1"]
lines = []

for encoding in encodings:
    try:
        with open("runs/audit_results/hard_examples.txt", encoding=encoding) as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        break
    except:
        continue

# 随机抽取50个
samples = random.sample(lines, min(50, len(lines)))

# 保存结果
with open("runs/target_review/hard_samples_spotcheck.txt", "w") as f:
    for sample in samples:
        f.write(f"{sample}\n")

print(f"已完成! 抽取了 {len(samples)} 个样本")
