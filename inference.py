import os
import cv2
import time
import torch
import numpy
import torch.nn as nn
from config import parse_args
from model.yolov1 import YOLOv1
import torch.nn.functional as F
from dataset.voc import VOCDataset
from dataset.augment import build_transform

def inference(args,
              dev,
              model,
              dataset):
    
    with torch.no_grad():
        for index in range(len(dataset)):
            start_time = time.time()

            image, label, ratio = dataset.__getitem__(index)
            image = image.unsqueeze(dim = 0).to(dev)

            outputs = model(image)

            ## origin image
            image, _ = dataset.pull_image(index)

            scores = outputs['scores']
            labels = outputs['labels']
            bboxes = outputs['bboxes']
            bboxes[..., [0, 2]] /= ratio[0]
            bboxes[..., [1, 3]] /= ratio[1]
            bboxes[..., [0, 2]] = numpy.clip(bboxes[..., [0, 2]], a_min=0., a_max=(image.shape[0]))
            bboxes[..., [1, 3]] = numpy.clip(bboxes[..., [1, 3]], a_min=0., a_max=(image.shape[1]))

            for i, box in enumerate(bboxes):
                score = scores[i]
                label = args.class_names[labels[i]]
                box = [int(point) for point in box]

                cv2.rectangle(image, (box[0], box[1]),  (box[2], box[3]), (0, 64, 255), 1)
                text = "%s:%s"%(label, str(round(float(score), 2)))
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                cv2.rectangle(image, (box[0], box[1]),  (box[0] + tw, box[1] + th), (0, 64, 255), -1) 
                cv2.putText(image, text, (box[0], box[1]+th), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            
           
            if args.real_time:
                fps = 1 / (time.time() - start_time)
                cv2.putText(image, 'FPS: '+str(round(fps,2)), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.imshow('image', image)

                # 退出循环的按键（通常是'q'键）  
                if cv2.waitKey(1) == ord('q'):  
                    break
            else:
                for label_id in range(args.num_classes):
                    ids = numpy.where(labels == label_id)[0]
                    if len(ids) != 0:
                        class_bboxes = bboxes[ids]
                        class_scores = scores[ids]
                        class_result_path = os.path.join(os.getcwd(), 'results', args.weight.replace('.pth', ''), args.class_names[label_id])
                        if not os.path.exists(class_result_path):
                            os.makedirs(class_result_path)

                        cv2.imwrite(os.path.join(class_result_path, dataset.ids[index][1]+'.jpg'), image)

                        filename = os.path.join(class_result_path, 'det_test_%s.txt'%(args.class_names[label_id]))
                        with open(filename, 'a+') as f:
                            for j, box in enumerate(class_bboxes):
                                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                                    dataset.ids[index][1], 
                                    class_scores[j],
                                    box[0], box[1], box[2], box[3]))
            
            print('Inference: {} / {}'.format(index+1, len(dataset)), end='\r')

if __name__ == "__main__":
    args = parse_args()
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # -------------------- Build Data --------------------
    val_transformer = build_transform(args, is_train = False)
    val_dataset = VOCDataset(
                            data_dir     = os.path.join(args.root, args.data),
                            image_sets   = args.val_sets,
                            transform    = val_transformer,
                            is_train     = False,
                            )
    
    # -------------------- Build Model --------------------
    model = YOLOv1(args = args, 
                   device = device,
                   trainable = False,
                   nms_thresh = args.nms_thresh,
                   conf_thresh = args.conf_thresh)
    weight_path = os.path.join(args.root, args.project, 'results', args.weight)
    checkpoint = torch.load(weight_path, map_location='cpu')
    checkpoint_state_dict = checkpoint["model"]
    model.load_state_dict(checkpoint_state_dict)
    model.to(device).eval()
    
    # --------------------- Start Inference --------------------
    all_boxes = [[[] for _ in range(len(val_dataset)) ] for _ in range(args.num_classes)]
    inference(
        args = args,
        dev = device,
        model = model,
        dataset = val_dataset)