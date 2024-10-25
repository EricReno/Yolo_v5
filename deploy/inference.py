import os
import cv2
import time
import numpy
import argparse
import onnxruntime
from xml.dom import minidom
import xml.etree.ElementTree as ET

fps = []

def parse_args():
    parser = argparse.ArgumentParser(description='Inference VOC20')
    parser.add_argument('--cuda', default=True, help='Use CUDA for inference.')
    parser.add_argument('--onnx', default='best.onnx', help='Path to the ONNX model file.')
    parser.add_argument('--image_size', default=512, type=int, help='Input image size.')
    parser.add_argument('--confidence', default=0.3, type=float, help='Confidence threshold for object detection.')
    parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold.')
    parser.add_argument('--class_names', nargs='+', default=['fire', 'smoke'], help='List of class names.')
    return parser.parse_args()

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


def setup_inference(args):
    providers = [('CUDAExecutionProvider', {'device_id': 0})] if args.cuda else [('CPUExecutionProvider', {})]
    print('Using CUDA' if args.cuda else 'Using CPU')
    return onnxruntime.InferenceSession(args.onnx, providers=providers)

def generate_colors(num_classes):
    numpy.random.seed(0)
    return [tuple(numpy.random.randint(255, size=3).tolist()) for _ in range(num_classes)]

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

def infer(input, session):   
    start = time.time()
    output = session.run(['output'], {'input': input})
    end = time.time()
    
    print(end-start)
    print("Inference FPS (Hz):", 1 / (end-start))

    fps.append( 1 / (end-start))

    return output

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

def main():
    args = parse_args()
    session = setup_inference(args)
    class_colors = generate_colors(len(args.class_names))
    cap = cv2.VideoCapture('video.mp4')

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
    
        start_time = time.time()
        image, infer_input, ratio = preinfer(image, args.image_size)
        postinfer_input = infer(infer_input, session)
        labels, scores, bboxes = postinfer(postinfer_input, ratio, args.image_size, args.class_names, args.confidence, args.nms_thresh)
        end_time = time.time()

        # save_results_to_xml(image_name, bboxes, labels, scores, args.class_names, image_size)

        # display_fps(image, end_time-start_time)
        draw_bboxes(image, bboxes, labels, scores, args.class_names, class_colors)

        cv2.imshow('image', image)
        
        cv2.waitKey(0)
    
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Mean fps:', numpy.mean(fps))

if __name__ == '__main__':
    main()