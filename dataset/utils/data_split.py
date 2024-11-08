import os
import random

# 文件夹路径
images_folder = 'JPEGImages'
annotations_folder = 'Annotations'

# 获取所有图像文件名
image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]

# 确保图像和标注文件匹配
image_files.sort()
annotation_files.sort()

missing_annotations = [f for f in image_files if f.replace('.jpg', '.xml') not in annotation_files]
if missing_annotations:
    print(f"Warning: Missing annotations for: {', '.join(missing_annotations)}")


# 设置随机种子，确保每次运行结果一致
random.seed(42)

# 计算数据集大小
total_files = len(image_files)
train_size = int(total_files * 0.9)  # 80% 用于训练
val_size = total_files - train_size  # 剩下的用于验证

# 随机打乱文件顺序
random.shuffle(image_files)

# 划分训练集和验证集
train_files = image_files[:train_size]
val_files = image_files[train_size:]

# 生成 train.txt 和 val.txt
with open('train.txt', 'w') as train_file:
    for file in train_files:
        train_file.write(file.split('.')[0] + '\n')

with open('val.txt', 'w') as val_file:
    for file in val_files:
        val_file.write(file.split('.')[0] + '\n')

print(f'Train and validation lists have been generated.')
