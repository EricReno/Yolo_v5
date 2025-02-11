import os
import cv2
import time
import argparse
from xml.dom import minidom
import xml.etree.ElementTree as ET
from inference import setup_inference, infer

def display_fps(image, time):
    fps = f"fps:{round(1 / (time), 2)}"
    cv2.putText(image, fps, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

def draw_bboxes(image, bboxes, labels, scores, class_names, class_colors):
    for index, bbox in enumerate(bboxes):
        bbox = [int(point) for point in bbox]

        text = "%s:%s"%(class_names[labels[index]], str(round(float(scores[index]), 2)))
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)

        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), class_colors[labels[index]], 2)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + w, bbox[1] + h), class_colors[labels[index]], -1) 
        cv2.putText(image, text, (bbox[0], bbox[1]+h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

def save_results_to_xml(args, label_path, bboxes, labels, scores, shape):                       
    root = ET.Element("annotations")
    
    ET.SubElement(root, "folder").text = os.path.dirname(label_path)
    ET.SubElement(root, "filename").text = label_path.split('//')[-1].replace('.xml', '.jpg')
    
    size_elem = ET.SubElement(root, "size")
    ET.SubElement(size_elem, "width").text = str(shape[1])
    ET.SubElement(size_elem, "height").text = str(shape[0])
    ET.SubElement(size_elem, "depth").text = str(shape[2])

    for bbox, label, score in zip(bboxes, labels, scores):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = args['class_names'][label]
        ET.SubElement(obj, "score").text = str(score)

        bbox_elem = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox_elem, "xmin").text = str(int(bbox[0]))
        ET.SubElement(bbox_elem, "ymin").text = str(int(bbox[1]))
        ET.SubElement(bbox_elem, "xmax").text = str(int(bbox[2]))
        ET.SubElement(bbox_elem, "ymax").text = str(int(bbox[3]))

    xml_str = ET.tostring(root, encoding='utf-8')
    xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")

    with open(label_path, "w") as f:
        f.write(xml_str)
        
def run(arg):
    session = setup_inference(args)
    
    if args.mode == 'image':
        # read a video
        files = [os.path.join(args.path_to_img, file) for file in os.listdir(args.path_to_img)]
        
        # for save
        if args.save:
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'{timestamp}.mp4', fourcc, fps, (int(width), int(height)))
        
        for file_path in files:
            image = cv2.imread(file_path)
            t0 = time.time()
            labels, scores, bboxes = infer(args, image, session)
            
            # label_path = file_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
            # save_results_to_xml(args, label_path, bboxes, labels, scores, image.shape)
            
            display_fps(image, time.time()-t0)
            draw_bboxes(image, bboxes, labels, scores, args.class_names, (255, 128, 0))
            
            if args.save:
                out.write(image)
            
            if args.show:
                show_img = cv2.resize(image, (1920, 1080))
                cv2.imshow('detection', show_img)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

        if args.save: out.release()
        cv2.destroyAllWindows()
    
    if args.mode == 'video':
        # read a video
        cap = cv2.VideoCapture(args.path_to_vid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # for save
        if args.save:
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'{timestamp}.mp4', fourcc, fps, (int(width), int(height)))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                t0 = time.time()
                labels, scores, bboxes = infer(args, frame, session)

                display_fps(frame, time.time()-t0)
                draw_bboxes(frame, bboxes, labels, scores, args.class_names, (255, 128, 0))
                
                if args.save:
                    out.write(frame)
                
                if args.show:
                    show_img = cv2.resize(frame, (1920, 1080))
                    cv2.imshow('detection', show_img)
                    ch = cv2.waitKey(0)
                    if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        break
            else:
                break
        cap.release()
        if args.save: out.release()
        cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Elevator')
    parser.add_argument('--cuda', default=True, help='Use CUDA for inference.')
    parser.add_argument('--mode', default='image', help=['video, image, camera'])
    parser.add_argument('--path_to_vid', default='video.mp4')
    parser.add_argument('--path_to_img', default='E:\BK\SV_Detection_Elevator\data\Private\JPEGImages')
    parser.add_argument('--show', default=True)
    parser.add_argument('--save', default=False)
    parser.add_argument('--onnx', default='elevator.onnx', help='Path to the ONNX model file.')
    parser.add_argument('--image_size', default=640, type=int, help='Input image size.')
    parser.add_argument('--confidence', default=0.5, type=float, help='Confidence threshold for object detection.')
    parser.add_argument('--nms_thresh', default=0.3, type=float, help='NMS threshold.')
    parser.add_argument('--class_names', default=['person', 'bicycle', 'motorcycle'], help='List of class names.')
    args = parser.parse_args()
    
    run(args)