import os 
import cv2
from xml.dom import minidom
import xml.etree.ElementTree as ET

def save2voc(labels, img_height, img_width, output_dir, id):
    annotation = ET.Element('annotation')
    
    ET.SubElement(annotation, 'filename').text = id+'.jpg'
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_width)
    ET.SubElement(size, 'height').text = str(img_height)
    
    for label in labels:
        label = label.strip().split()
        class_id = int(label[0])  # 类别 ID
        track_id = int(label[1])  # 跟踪 ID
        center_x = float(label[2]) * img_width
        center_y = float(label[3]) * img_height
        width = float(label[4]) * img_width
        height = float(label[5]) * img_height
        
        x_min = int(center_x - width / 2)
        y_min = int(center_y - height / 2)
        x_max = int(center_x + width / 2)
        y_max = int(center_y + height / 2)

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = str(class_id)
        ET.SubElement(obj, 'track_id').text = str(track_id)
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(x_min)
        ET.SubElement(bndbox, 'ymin').text = str(y_min)
        ET.SubElement(bndbox, 'xmax').text = str(x_max)
        ET.SubElement(bndbox, 'ymax').text = str(y_max)

    xml_str = ET.tostring(annotation, encoding='utf-8', xml_declaration=True)
    parsed_xml = minidom.parseString(xml_str)
    formatted_xml = parsed_xml.toprettyxml(indent="    ")

    xml_file_path = os.path.join(output_dir, f"{id}.xml")
    with open(xml_file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_xml)

    print(f"XML 文件已保存至 {output_dir}/{id}.xml")

def yolo2voc(images_dir, src, dist):
    for label_file in os.listdir(src):
        id = label_file.split('.')[0]
        
        image = cv2.imread(os.path.join(images_dir, id+'.jpg'))
        i_h, i_w = image.shape[:2]

        with open(os.path.join(src, label_file), 'r') as f:
            labels = f.readlines()
            save2voc(labels, i_h, i_w, dist, id)

if __name__ == '__main__':
    images_dir = "E:\data\PRW\JPEGImages"
    labels_dir = "E:\data\PRW\Annotations"
    n_labels_dir = "E:\data\PRW\Annotations_n"

    yolo2voc(images_dir, labels_dir, n_labels_dir)