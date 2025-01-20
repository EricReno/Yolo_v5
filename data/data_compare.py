import os
import hashlib
from PIL import Image

def calculate_image_hash(image_path):
    """计算图片的哈希值"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # 确保一致性
        img_data = img.tobytes()
        return hashlib.md5(img_data).hexdigest()

def find_duplicate_images(folder_path):
    """查找文件夹下内容相同但名字不同的图片"""
    hash_map = {}  # 用于存储哈希值和文件路径
    duplicates = []  # 用于存储重复的图片对

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_hash = calculate_image_hash(file_path)
                if file_hash in hash_map:
                    duplicates.append((hash_map[file_hash], file_path))
                else:
                    hash_map[file_hash] = file_path
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return duplicates

if __name__ == "__main__":
    folder = "JPEGImages"  # 替换为你的目标文件夹路径
    duplicates = find_duplicate_images(folder)
    
    abc = []

    if duplicates:
        print("Found duplicate images:")
        for original, duplicate in duplicates:
            if original not in abc:
                abc.append(original)
                abc.append(duplicate)
                print(f"Duplicate: {duplicate} is the same as {original}")
    else:
        print("No duplicate images found.")
