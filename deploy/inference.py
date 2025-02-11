import onnxruntime
import numpy as np
from PIL import Image

def setup_inference(args):
    providers = [('CUDAExecutionProvider', {'device_id': 0})] if args.cuda else [('CPUExecutionProvider', {})]
    print('Using CUDA' if args.cuda else 'Using CPU')

    return onnxruntime.InferenceSession(args.onnx, providers=providers)

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
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def preinfer(image, image_size):
    ratio = [image_size/image.shape[1], image_size/image.shape[0]]

    image_pil = Image.fromarray(image)
    output_pil = image_pil.resize((image_size, image_size))
    output = np.array(output_pil)
    
    output = output.transpose([2, 0, 1]).astype(np.float32)
    output /= 255.
    output = np.expand_dims(output, 0)

    return  image, output, ratio

def postinfer(input, ratio, image_size, class_names, conf_thresh, nms_thresh):
    bboxes = input[0][:, :4]
    scores = input[0][:, 4:]

    labels = np.argmax(scores, axis=1)
    scores = scores[(np.arange(scores.shape[0]), labels)]
        
    # 初筛: confidecn threshold 
    keep = np.where(scores >= conf_thresh)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # 放缩：放缩到原图&边界值处理
    bboxes[..., [0, 2]] /= ratio[0]
    bboxes[..., [1, 3]] /= ratio[1]
    bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=(image_size/ratio[0]))
    bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=(image_size/ratio[1]))

    # NMS: Non-Maximum Suppression
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(len(class_names)):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1
    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return labels, scores, bboxes

def infer(args, image, session):
    image, infer_input, ratio = preinfer(image, args.image_size)
    postinfer_input = session.run(['output'], {'input': infer_input})
    labels, scores, bboxes = postinfer(postinfer_input, 
                                       ratio, 
                                       args.image_size, 
                                       args.class_names, 
                                       args.confidence, 
                                       args.nms_thresh
                                       )

    return labels, scores, bboxes