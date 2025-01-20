import os
import re
import shutil

jpegimages_dir = 'JPEGImages'
annotations_dir = 'Annotations'
target_jpegimages_dir = 'JPEG'
target_annotations_dir = 'Anno'

os.makedirs(target_jpegimages_dir, exist_ok=True)
os.makedirs(target_annotations_dir, exist_ok=True)

def natural_sort_key(s):
    # 提取字符串中的数字部分
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

for index, image in enumerate(sorted(os.listdir(jpegimages_dir), key=natural_sort_key)):
    source_path = os.path.join(jpegimages_dir, image)
    new_name = f"{index + 3271:06d}.jpg"  # 带编号的文件名
    target_path = os.path.join(target_jpegimages_dir, new_name)
    shutil.copy(source_path, target_path)

    source_path = os.path.join(annotations_dir, image.replace('.jpg', '.xml'))
    new_name = f"{index + 3271:06d}.xml"  # 带编号的文件名
    target_path = os.path.join(target_annotations_dir, new_name)
    shutil.copy(source_path, target_path)
    
    print(f"Processing {index}/{len(os.listdir(jpegimages_dir))}", end='\r')

os.rename(jpegimages_dir, jpegimages_dir + '_bk')
os.rename(annotations_dir, annotations_dir + '_bk')

os.rename(target_jpegimages_dir, jpegimages_dir)
os.rename(target_annotations_dir, annotations_dir)

print(f"Files have been copied and renamed in '{target_jpegimages_dir}' directory.")