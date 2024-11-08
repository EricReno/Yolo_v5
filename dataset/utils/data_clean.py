import os
images_list = os.listdir('JPEGImages')

cnt = 0
for xml_name in os.listdir('Annotations'):
    if xml_name.split('.')[0]+'.jpg' not in images_list:
        os.remove(os.path.join('Annotations', xml_name))
        cnt += 1
    
print(cnt)

