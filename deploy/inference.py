import os
import cv2
import time
import numpy
import argparse
import onnxruntime
from xml.dom import minidom
import xml.etree.ElementTree as ET

def save_results_to_xml(image_name, bboxes, labels, scores, class_names, image_size):
    root = ET.Element("annotations")
    
    # 添加 folder 和 filename
    folder_elem = ET.SubElement(root, "folder").text = "E:\\data\\FireDetection\\Annotations"
    filename_elem = ET.SubElement(root, "filename").text = image_name
    
    # 添加 size 元素
    size_elem = ET.SubElement(root, "size")
    ET.SubElement(size_elem, "width").text = str(image_size[0])
    ET.SubElement(size_elem, "height").text = str(image_size[1])
    ET.SubElement(size_elem, "depth").text = "3"  # Assuming RGB images


    for bbox, label, score in zip(bboxes, labels, scores):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_names[label]
        ET.SubElement(obj, "score").text = str(score)

        bbox_elem = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox_elem, "xmin").text = str(int(bbox[0]))
        ET.SubElement(bbox_elem, "ymin").text = str(int(bbox[1]))
        ET.SubElement(bbox_elem, "xmax").text = str(int(bbox[2]))
        ET.SubElement(bbox_elem, "ymax").text = str(int(bbox[3]))

     # Convert the tree to a string
    xml_str = ET.tostring(root, encoding='utf-8')

    # Format the XML string
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")

    xml_file_name = os.path.splitext(image_name)[0] + ".xml"

    with open(os.path.join("OutputXML", xml_file_name), "w") as f:
        f.write(pretty_xml_str)

def display_fps(image, time):
    fps = f"fps:{round(1 / (time), 2)}"
    cv2.putText(image, fps, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

def draw_bboxes(image, bboxes, labels, scores, class_names, class_colors):
    for index, bbox in enumerate(bboxes):
        bbox = [int(point) for point in bbox]

        text = "%s:%s"%(class_names[labels[index]], str(round(float(scores[index]), 2)))
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)

        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), class_colors[labels[index]])
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + w, bbox[1] + h), class_colors[labels[index]], -1) 
        cv2.putText(image, text, (bbox[0], bbox[1]+h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = numpy.maximum(x1[i], x1[order[1:]])
        yy1 = numpy.maximum(y1[i], y1[order[1:]])
        xx2 = numpy.minimum(x2[i], x2[order[1:]])
        yy2 = numpy.minimum(y2[i], y2[order[1:]])

        w = numpy.maximum(1e-10, xx2 - xx1)
        h = numpy.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = numpy.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def preinfer(image, image_size):
    ratio = [image_size/image.shape[1], image_size/image.shape[0]]

    output = cv2.resize(image, (image_size, image_size))
    output = output.transpose([2, 0, 1]).astype(numpy.float32)
    output /= 255.
    output = numpy.expand_dims(output, 0)

    return  image, output, ratio

def postinfer(input, ratio, image_size, class_names, conf_thresh, nms_thresh):
    bboxes = input[0][:, :4]
    scores = input[0][:, 4:]

    labels = numpy.argmax(scores, axis=1)
    scores = scores[(numpy.arange(scores.shape[0]), labels)]
        
    # 初筛: confidecn threshold 
    keep = numpy.where(scores >= conf_thresh)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # 放缩：放缩到原图&边界值处理
    bboxes[..., [0, 2]] /= ratio[0]
    bboxes[..., [1, 3]] /= ratio[1]
    bboxes[..., [0, 2]] = numpy.clip(bboxes[..., [0, 2]], a_min=0., a_max=(image_size/ratio[0]))
    bboxes[..., [1, 3]] = numpy.clip(bboxes[..., [1, 3]], a_min=0., a_max=(image_size/ratio[1]))

    # NMS: Non-Maximum Suppression
    keep = numpy.zeros(len(bboxes), dtype=numpy.int32)
    for i in range(len(class_names)):
        inds = numpy.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1
    keep = numpy.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return labels, scores, bboxes

def setup_inference(args):
    providers = [('CUDAExecutionProvider', {'device_id': 0})] if args.cuda else [('CPUExecutionProvider', {})]
    print('Using CUDA' if args.cuda else 'Using CPU')

    return onnxruntime.InferenceSession(args.onnx, providers=providers)

def run(args):
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
            frame = cv2.imread(file_path)
            frame, input, ratio = preinfer(frame, args.image_size)
            t0 = time.time()
            infer = session.run(['output'], {'input': input})
            labels, scores, bboxes = postinfer(infer, ratio, args.image_size, args.class_names, args.confidence, args.nms_thresh)

            display_fps(frame, time.time()-t0)
            draw_bboxes(frame, bboxes, labels, scores, args.class_names, (255, 255, 0))
            
            if args.save:
                out.write(frame)
            
            if args.show:
                show_img = cv2.resize(frame, (1920, 1080))
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
                frame, input, ratio = preinfer(frame, args.image_size)
                t0 = time.time()
                infer = session.run(['output'], {'input': input})
                labels, scores, bboxes = postinfer(infer, ratio, args.image_size, args.class_names, args.confidence, args.nms_thresh)

                display_fps(frame, time.time()-t0)
                draw_bboxes(frame, bboxes, labels, scores, args.class_names, (255, 255, 0))
                
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
    parser = argparse.ArgumentParser(description='Inference Plate')
    parser.add_argument('--cuda', default=True, help='Use CUDA for inference.')
    parser.add_argument('--mode', default='video', help=['video, image, camera'])
    parser.add_argument('--path_to_vid', default='plate.mp4')
    parser.add_argument('--path_to_img', default='D:\SV_Person_Detection\data\BK\\bk\data_realpeople\JPEGImages')
    parser.add_argument('--show', default=True)
    parser.add_argument('--save', default=False)
    parser.add_argument('--onnx', default='plate.onnx', help='Path to the ONNX model file.')
    parser.add_argument('--image_size', default=640, type=int, help='Input image size.')
    parser.add_argument('--confidence', default=0.7, type=float, help='Confidence threshold for object detection.')
    parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold.')
    parser.add_argument('--class_names', default=['plate'], help='List of class names.')
    args = parser.parse_args()
    
    run(args)