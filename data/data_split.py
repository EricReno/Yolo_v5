import os
import random

def data_split(data_root, images_folder, annotations_folder):
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]

    image_files.sort()
    annotation_files.sort()

    missing_annotations = [f for f in image_files if f.replace('.jpg', '.xml') not in annotation_files]
    if missing_annotations:
        print(f"Warning: Missing annotations for: {', '.join(missing_annotations)}")

    random.seed(42)

    total_files = len(image_files)
    train_size = int(total_files * 0.95)
    val_size = total_files - train_size
    
    random.shuffle(image_files)

    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    with open(os.path.join(data_root, 'train.txt'), 'w') as train_file:
        for file in train_files:
            train_file.write(file.split('.')[0] + '\n')

    with open(os.path.join(data_root, 'val.txt'), 'w') as val_file:
        for file in val_files:
            val_file.write(file.split('.')[0] + '\n')

    print(f'Train and validation lists have been generated in {data_root}.')

if __name__ == "__main__":
    data_list = ['CCPD2019', 'CCPD2020', 'CCPD2021']
    for data_root in data_list:
        for _, folders, files in os.walk(data_root):
            if 'JPEGImages' in folders and 'Annotations' in folders:
                JPEGImages = os.path.join(_, 'JPEGImages')
                Annotations = os.path.join(_, 'Annotations')
                
                data_split(_, JPEGImages, Annotations)