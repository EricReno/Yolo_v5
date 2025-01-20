import os
import xml.etree.ElementTree as ET  

os.makedirs('Anno', exist_ok=True)

for xml in os.listdir('Annotations'):
    xml_path = os.path.join('Annotations', xml)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name != 'person':
            root.remove(obj)
    
    output_path = os.path.join('Anno', xml)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"Processed {xml} -> {output_path}")