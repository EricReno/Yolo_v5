# import cv2
# import sys
# import time
# import numpy
# sys.path.append('../')
# from config import parse_args

# cap = cv2.VideoCapture('video.mp4')

# ort_session = onnxruntime.InferenceSession("yolo_v1.onnx")

# classnames = [  'aeroplane', 'bicycle',   'bird',      'boat',    'bottle', 
#                       'bus',     'car',    'cat',     'chair',       'cow', 
#               'diningtable',     'dog',  'horse', 'motorbike',    'person', 
#               'pottedplant',   'sheep',   'sofa',     'train', 'tvmonitor']

# frame_count = 0  
# last_time = time.time() 

# args = parse_args()

# import numpy as np

# ## basic NMS
# def nms(bboxes, scores, nms_thresh):
#     """"Pure Python NMS."""
#     x1 = bboxes[:, 0]  #xmin
#     y1 = bboxes[:, 1]  #ymin
#     x2 = bboxes[:, 2]  #xmax
#     y2 = bboxes[:, 3]  #ymax

#     areas = (x2 - x1) * (y2 - y1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         # compute iou
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(1e-10, xx2 - xx1)
#         h = np.maximum(1e-10, yy2 - yy1)
#         inter = w * h

#         iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
#         #reserve all the boundingbox whose ovr less than thresh
#         inds = np.where(iou <= nms_thresh)[0]
#         order = order[inds + 1]

#     return keep

# def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
#     # nms
#     keep = np.zeros(len(bboxes), dtype=np.int32)
#     for i in range(num_classes):
#         inds = np.where(labels == i)[0]
#         if len(inds) == 0:
#             continue
#         c_bboxes = bboxes[inds]
#         c_scores = scores[inds]
#         c_keep = nms(c_bboxes, c_scores, nms_thresh)
#         keep[inds[c_keep]] = 1
#     keep = np.where(keep > 0)
#     scores = scores[keep]
#     labels = labels[keep]
#     bboxes = bboxes[keep]

#     return scores, labels, bboxes

# def postprocess(bboxes, scores):
#     """
#     Input:
#         bboxes: [HxW, 4]
#         scores: [HxW, num_classes]
#     Output:
#         bboxes: [N, 4]
#         score:  [N,]
#         labels: [N,]
#     """

#     labels = np.argmax(scores, axis=1)

#     scores = scores[(np.arange(scores.shape[0]), labels)]
    
#     # threshold
#     keep = np.where(scores >= args.conf_thresh)
#     bboxes = bboxes[keep]
#     scores = scores[keep]
#     labels = labels[keep]
#     print(bboxes)
#     print(scores)

#     print(labels)


#     # nms
#     scores, labels, bboxes = multiclass_nms_class_aware(
#         scores, labels, bboxes, args.nms_thresh, args.num_classes)

#     return bboxes, scores, labels

# while True:
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.resize(frame, (640, 640))
        
#         input = frame.astype(numpy.float32)
#         input = input.transpose([2, 0, 1])
#         input = numpy.expand_dims(input, 0)

#         ort_inputs = {'input': input}
#         ort_output = ort_session.run(['output'], ort_inputs)[0]

#         bboxes = ort_output[:, :4]
#         scores = ort_output[:, 4:]
#         bboxes, scores, labels = postprocess(bboxes, scores)


        

#         for index, id in enumerate(labels):        
#             # text2 = "pred:%s"%(classnames[id])
#             # (text_width, text_height), baseline = cv2.getTextSize(text2, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
#             # cv2.rectangle(frame, (int(bboxes[index][0]), int(bboxes[index][1])),  (10 + text_width, 10 + text_height), (0, 128, 255), -1) 
#             # cv2.putText(frame, text2, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#             print(bboxes[index])
#             cv2.rectangle(frame, (int(bboxes[index][0]), int(bboxes[index][1])),  (int(bboxes[index][2]), int(bboxes[index][3])), (0, 128, 255), 1) 

#             # frame_count += 1  
#             # elapsed_time = time.time() - last_time  
#             # fps = frame_count / elapsed_time  
#             # cv2.putText(frame, 'FPS: '+str(round(fps,2)), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            
#         #     cv2.imshow('1', frame)
#         #     cv2.waitKey(0)
#         # break

#         cv2.imshow('image', frame)

#         if cv2.waitKey(1) == ord('q'):  
#             break
#     else:
#         break

# cv2.destroyAllWindows()

import os
import cv2
import numpy
import argparse
import onnxruntime


def parse_args():
    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument('--onnx', default='yolo_v1.onnx', help='The onnx file which will be used.')
    parser.add_argument('--image_path', default='images', help='The root directory where data are stored')
    parser.add_argument('--image_size', default= 640,     help='input image size')

    parser.add_argument('--confidece', default=0.3,       help='The confidence threshold of predicted objects')
    parser.add_argument('--class_names', default= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
                           'sofa', 'train', 'tvmonitor'], help= 'The category of predictions that the model can cover')
    

    parser.add_argument('--cuda', default=True, help='Weather use cuda.')
    # parser.add_argument('--root', default='/data/ryb/', help='The root directory where code and data are stored')
    parser.add_argument('--root', default='E://', help='The root directory where code and data are stored')
    parser.add_argument('--data', default='data/VOCdevkit', help='The path where the dataset is stored')
    parser.add_argument('--project', default='ObjectDetection_VOC20', help='The path where the project code is stored')
    parser.add_argument('--print_frequency', default=10, type=int, help='The print frequency')

    # data & model
    parser.add_argument('--num_workers', default=16, help='epoch for warm_up')
    parser.add_argument('--val_sets',   default=[('2007', 'test')], help='The data set to be tested')
    parser.add_argument('--train_sets', default=[('2007', 'trainval'), ('2012', 'trainval')], help='The data set to be trained')
    parser.add_argument('--num_classes', default=20, help='The number of the classes')
   
    parser.add_argument('--backbone', default='resnet18', help=['resnet18', 'resnet34'])
    parser.add_argument('--expand_ratio', default=0.5, help='The expand_ratio')
    parser.add_argument('--pooling_size', default=5, help='The pooling size setting')
    parser.add_argument('--loss_obj_weight', default=1.0, help='The number of the classes')
    parser.add_argument('--loss_cls_weight', default=1.0, help='The number of the classes')
    parser.add_argument('--loss_box_weight', default=5.0, help='The number of the classes')
    
    """
    Train configuration
    """
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('--lr_momentum', default=0.937, help='lr_momentum')
    parser.add_argument('--lr_weight_decay', default=0.0005, help='lr_weight_decay')
    parser.add_argument('--warmup_epoch', default=3, help='epoch for warm_up')
    parser.add_argument('--warmup_momentum', default=0.8, help='epoch for warm_up')
    parser.add_argument('--grad_accumulate', default=1, type=int, help='gradient accumulation')
    parser.add_argument('--resume', default='34.pth', type=str, help=['None','44.pth'])
    parser.add_argument('--batch_size', default=12, help='The batch size used by a single GPU during training')
    parser.add_argument('--save_folder', default='results', help='The path for wights')
    parser.add_argument('--max_epoch', default=135, help='The maximum epoch used in this training')
    parser.add_argument('--save_epoch', default=0, help='The epoch when the model parameters are saved')
    parser.add_argument('--pretrained', default=True, help='Whether to use pre-training weights')
    parser.add_argument('--data_augmentation', default=['RandomSaturationHue', 'RandomContrast', 'RandomBrightness', 'RandomSampleCrop', 'RandomExpand', 'RandomHorizontalFlip'],
                        help="[RandomExpand,RandomCenterCropPad,RandomCenterCropPad, RandomBrightness], default Resize")
    parser.add_argument('--ema', action='store_true', default=False, help='Model EMA')
    """
    Evaluate configuration
    """
    parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold')
    parser.add_argument('--recall_thr', default=101, help='The threshold for recall')
    parser.add_argument('--eval_weight', default='82.pth', type=str, help="Trained state_dict file path")
    parser.add_argument('--threshold', default=0.5, help='The iou threshold')
    parser.add_argument('--real_time', default=True, help='whether to real-time display detection results')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False, help='fuse Conv & BN')

    return parser.parse_args()

def preinfer(image_path, image_size):
    image = cv2.imread(image_path)

    ratio = [image_size/image.shape[1], image_size/image.shape[0]]

    output = cv2.resize(image, (image_size, image_size))
    output = output.astype(numpy.float32)
    output = output.transpose([2, 0, 1])
    output = numpy.expand_dims(output, 0)

    return  image, output, ratio

def infer(input, onnx):
    session = onnxruntime.InferenceSession(onnx)

    output = session.run(['output'], {'input': input})

    return output

def postinfer(input, conf_thresh, ratio, image):
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
    bboxes[..., [0, 2]] = numpy.clip(bboxes[..., [0, 2]], a_min=0., a_max=(image.shape[0]))
    bboxes[..., [1, 3]] = numpy.clip(bboxes[..., [1, 3]], a_min=0., a_max=(image.shape[1]))
    print(len(bboxes))
    print(bboxes)
    return labels, scores, bboxes


        # # nms
        # scores, labels, bboxes = self.multiclass_nms_class_aware(
        #     scores, labels, bboxes, self.nms_thresh, self.num_classes)

        # return bboxes, scores, labels





    

if __name__ == "__main__":
    args = parse_args()

    images_list = [os.path.join(os.path.abspath(args.image_path), _) for _ in os.listdir(args.image_path)]
    for image in images_list:

        ## TODO ONE
        image, infer_input, ratio = preinfer(image, args.image_size)

        ## TODO TWO
        postinfer_input = infer(infer_input, args.onnx) # 400*(4+20)

        ## TODO THREE
        labels, scores, bboxes = postinfer(postinfer_input, args.confidece, ratio, image)

        ## TODO FOUR
        for i, bbox in enumerate(bboxes):
            score = scores[i]
            label = labels[i]
            
            label_name = args.class_names[label]
            bbox = [int(point) for point in bbox]


            cv2.rectangle(image, (bbox[0], bbox[1]),  (bbox[2], bbox[3]), (0, 64, 255), 1)
            # text = "%s:%s"%(label, str(round(float(score), 2)))
            # (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            # cv2.rectangle(image, (bbox[0], bbox[1]),  (bbox[0] + tw, bbox[1] + th), (0, 64, 255), -1) 
            # cv2.putText(image, text, (bbox[0], bbox[1]+th), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            
            # if args.real_time:
            #     fps = 1 / (time.time() - startq_time)
            #     cv2.putText(image, 'FPS: '+str(round(fps,2)), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            #     cv2.imshow('image', image)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        
        # import time
        # time.sleep(1)

        # # 退出循环的按键（通常是'q'键）  
        if cv2.waitKey(1) == ord('q'):  
            break
