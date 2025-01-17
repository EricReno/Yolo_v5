import os
import cv2
import torch
import shutil
import numpy as np
from config import parse_args
from model.build import build_yolo
from dataset.build import build_transform, build_dataset

def rescale_bboxes(bboxes, origin_size, ratio):
    # rescale bboxes
    if isinstance(ratio, float):
        bboxes /= ratio
    elif isinstance(ratio, list):
        bboxes[..., [0, 2]] /= ratio[0]
        bboxes[..., [1, 3]] /= ratio[1]
        
    bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=origin_size[0])
    bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=origin_size[1])

    return bboxes

class Evaluator():
    """ VOC AP Evaluation class"""
    def __init__(self,
                 device,
                 dataset,
                 ovthresh,
                 class_names,
                 recall_thre,
                 visualization) -> None:
        
        self.device = device
        self.dataset = dataset
        self.ovthresh = ovthresh
        self.recall_thre = recall_thre
        self.class_names = class_names
        self.visualization = visualization
        self.num_classes = len(class_names)
        
        self.all_gt_boxes = [{} for _ in range(self.num_classes)]
        self.all_det_boxes = [[[] for _ in range(len(self.dataset))
                              ] for _ in range(self.num_classes)]
        # all_det_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)

    # def voc_ap(self, recall, precision):
    #     """ 
    #     Compute the average precision, given the recall and precision curves.
    #     Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        
    #     # Arguments
    #         recall:    The recall curve (np.array).
    #         precision: The precision curve (np.array).
    #     # Returns
    #         The average precision as computed in py-faster-rcnn.
    #     """
        
    #     # first append sentinel values at the end
    #     mrec = np.concatenate(([0.0], recall, [1.0]))
    #     mpre = np.concatenate(([1.0], precision, [0.0]))
        
    #     # compute the precision envelope
    #     for i in range(mpre.size - 1, 0, -1):
    #         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            
    #     # where X axis (recall) changes value
    #     i = np.where(mrec[:-1] != mrec[1:])[0]     

    #     # and sum (\Delta recall) * prec
    #     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    #     return ap
    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap
  
    def inference(self, model):
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video = cv2.VideoWriter('result.mp4', fourcc, 1, (int(1000*0.8), int(415*0.8)))
        for i in range(len(self.dataset)):
            img, _ = self.dataset.pull_image(i)
            orig_h, orig_w = img.shape[:2]

            image, target, deltas = self.dataset[i]

            image = image.unsqueeze(0).to(self.device)

            outputs = model(image)
            scores = outputs['scores']
            labels = outputs['labels']
            bboxes = outputs['bboxes']

            # rescale prediction bboxes
            bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], deltas)
            for j in range(self.num_classes):
                inds = np.where(labels == j)[0]
                if len(inds) == 0:
                    self.all_det_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                self.all_det_boxes[j][i] = c_dets

            # rescale grountruth targets
            target['boxes'] = rescale_bboxes(target['boxes'], [orig_w, orig_h], deltas)
            for j in range(self.num_classes):
                inds = np.where((target['labels'] == j) | (target['labels'] == j+0.1))[0]

                if len(inds) == 0:
                    self.all_gt_boxes[j][self.dataset.ids[i][1]] = np.array([], dtype=np.float32)
                    continue
                c_gt_bboxes = target['boxes'][inds]

                frac, _ = np.modf(target['labels'][inds])
                c_gt_diffcult = np.where(np.round(frac * 10) == 1, 1, 0)

                c_gts = np.hstack((c_gt_bboxes, c_gt_diffcult[:, np.newaxis])).astype(np.float32, copy=False)
                self.all_gt_boxes[j][self.dataset.ids[i][1]] = c_gts
            
            if self.visualization:
                # TODO  Visualization Debug
                if len(bboxes) == 0 and len(target['boxes']) == 0:
                    continue
                else:
                    import cv2
                    np.random.seed(0)
                    class_colors = [(np.random.randint(255),
                                    np.random.randint(255),
                                    np.random.randint(255)) for _ in range(self.num_classes)]

                    show_image, _ = self.dataset.pull_image(i)
                    prediction_image = show_image.copy()
                    groundtruth_image = show_image.copy()
                    
                    # prediction
                    for index, box in enumerate(bboxes):
                        text = "%s:%s"%(self.class_names[labels[index]], str(round(float(scores[index]), 2)))
                        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        cv2.rectangle(prediction_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), class_colors[labels[index]], int(box[2]-box[0])//1000)
                        cv2.rectangle(prediction_image, (int(box[0]), int(box[1])),  (int(box[0]) + w, int(box[1]) + h), class_colors[labels[index]], -1) 
                        cv2.putText(prediction_image, text, (int(box[0]), int(box[1])+h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    text_image = np.zeros((40, prediction_image.shape[1], 3), dtype=np.uint8)
                    cv2.putText(text_image, 'Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    prediction_image = np.concatenate((prediction_image, text_image), axis=0)

                    # groundtruth
                    for index, box in enumerate(target['boxes']):
                        text = self.class_names[int(target['labels'][index])]
                        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        cv2.rectangle(groundtruth_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), class_colors[int(target['labels'][index])], int(box[2]-box[0])//1000)
                        cv2.rectangle(groundtruth_image, (int(box[0]), int(box[1])),  (int(box[0]) + w, int(box[1]) + h), class_colors[int(target['labels'][index])], -1) 
                        cv2.putText(groundtruth_image, text, (int(box[0]), int(box[1])+h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    text_image = np.zeros((40, groundtruth_image.shape[1], 3), dtype=np.uint8)
                    cv2.putText(text_image, 'GroundTruth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    groundtruth_image = np.concatenate((groundtruth_image, text_image), axis=0)

                    show_image = np.concatenate((groundtruth_image, prediction_image), axis=1)
                    show_image = cv2.resize(show_image, (1920, 580))
                    # cv2.imshow('1', show_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # show_image = cv2.resize(show_image, (int(1000*0.8), int(415*0.8)))   
                    # video.write(show_image)

            print('Inference: {} / {}'.format(i+1, len(self.dataset)), end='\r')

        # video.release()
        return show_image
        
    def load_gt(self, classname):
        npos = 0
        gts = {}

        class_index = self.class_names.index(classname)
        
        for image_id in self.all_gt_boxes[class_index]:
            if len(self.all_gt_boxes[class_index][image_id]) == 0:
                gts[image_id] = {}
                continue

            difficult = np.array([box[-1] for box in self.all_gt_boxes[class_index][image_id]]).astype(bool)
            gts[image_id] = {'bbox': np.array(self.all_gt_boxes[class_index][image_id][:, :4]),
                             'difficult': difficult,
                             'det': [False] * len(self.all_gt_boxes[class_index][image_id])}
            
            npos = npos + sum(~difficult)
        
        return gts, npos

    def load_dets(self, classname):
        image_ids = []
        confidence = []
        bboxes = []

        class_index = self.class_names.index(classname)
        for im_ind, dets in enumerate(self.all_det_boxes[class_index]):
            image_id = self.dataset.ids[im_ind][1]
            for k in range(dets.shape[0]):
                image_ids.append(image_id)
                confidence.append(dets[k, -1])
                bboxes.append([dets[k, 0]+1, dets[k, 1]+1, dets[k, 2]+1, dets[k, 3]+1])
       
        return {
            'image_ids': np.array(image_ids),
            'confidence': np.array(confidence),
            'bboxes': np.array(bboxes)
        }

    def eval(self, model):
        show_image = self.inference(model)
        print('\n~~~~~~~~')
        print('Results:')

        aps = []
        pr = {}
        for cls_ind, cls_name in enumerate(self.class_names):
            dets = self.load_dets(cls_name)
            gts, npos = self.load_gt(cls_name)

            if len(dets['bboxes']):
                sorted_index = np.argsort(-dets['confidence'])
                sorted_image_ids = dets['image_ids'][sorted_index]
                sorted_confidence = dets['confidence'][sorted_index]
                sorted_bboxes = dets['bboxes'][sorted_index, :].astype(float)
                
                tp = np.zeros(len(dets['bboxes']))
                fp = np.zeros(len(dets['bboxes']))

                for index, box in enumerate(sorted_bboxes):
                    gt_dic = gts[sorted_image_ids[index]]
                    if 'bbox' in gt_dic:
                        gt_boxes = gt_dic['bbox'].astype(float)
                        x_min = np.maximum(gt_boxes[:, 0], box[0])
                        y_min = np.maximum(gt_boxes[:, 1], box[1])
                        x_max = np.minimum(gt_boxes[:, 2], box[2])
                        y_max = np.minimum(gt_boxes[:, 3], box[3])

                        w_intersect = np.maximum(x_max - x_min, 0.)
                        h_intersect = np.maximum(y_max - y_min, 0.)

                        dt_area = (box[2] - box[0]) * (box[3] - box[1])
                        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

                        area_intersect = w_intersect * h_intersect
                        area_union = gt_area + dt_area - area_intersect
                        ious = area_intersect / np.maximum(area_union, 1e-10)

                        max_iou, max_index = np.max(ious), np.argmax(ious)

                        if max_iou > self.ovthresh:
                            if not gt_dic['difficult'][max_index]:
                                if not gt_dic['det'][max_index]:
                                    tp[index] = 1
                                    gt_dic['det'][max_index] = 1
                                else:
                                    fp[index] = 1 
                        else:
                            fp[index] = 1 
                    else:
                        fp[index] = 1 

                # compute precision recall
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)

                rec = tp / float(npos)
                # avoid divide by zero in case the first detection matches a difficult
                # ground truth
                tp = np.nan_to_num(tp, nan=0.0)
                fp = np.nan_to_num(fp, nan=0.0)
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
                ap = self.voc_ap(rec, prec)
                pr[cls_name] = prec[-1]

                # ## 插值的P-R曲线
                # rec_interp=np.linspace(0, 1, self.recall_thre) #101steps, from 0% to 100% 
                # prec = np.interp(rec_interp, rec, prec, right=0)

                # ap = self.voc_ap(rec_interp, prec)
            else:
                rec = 0.
                prec = 0.
                ap = 0.

            aps += [ap]
            
            print('{:<12} :     {:.3f}'.format(cls_name, ap))

        self.map = np.mean(aps)
        print('')
        print('~~~~~~~~')
        print('Mean AP = {:.4f}%'.format(np.mean(aps)*100))
        print('~~~~~~~~')
        print('')

        return self.map, show_image

def build_eval(args, dataset, device):
    evaluator = Evaluator(
        device   =device,
        dataset  = dataset,
        ovthresh = args.eval_ovthresh,                        
        class_names = args.class_names,
        recall_thre = args.recall_threshold,
        visualization = args.eval_visualization)
    
    return evaluator
    
if __name__ == "__main__":
    os.makedirs('D:\SV_Person_Detection\data\TMP\JPEGImages', exist_ok=True)
    os.makedirs('D:\SV_Person_Detection\data\TMP\Annotations', exist_ok=True)
    os.makedirs('D:\SV_Person_Detection\data\TMP\Visualization', exist_ok=True)
    
    threshold = 0.5
    args = parse_args()
    args.nms_threshold=0.3
    args.confidence_threshold = 0.5
    args.eval_visualization = True
    args.resume_weight_path = "None"
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('use cuda')
    else:
        device = torch.device('cpu')
        
    model = build_yolo(args, device, False)
    model = model.eval()

    state_dict = torch.load(f = os.path.join('deploy', args.model_weight_path), 
                            map_location = 'cpu', weights_only = False)
    model.load_state_dict(state_dict["model"])
    print('epoch:', state_dict['epoch'])
    print('mAP:', state_dict['mAP'])
        
    datadir = "D:\SV_Person_Detection\data\Public"
    images_folder = os.path.join(datadir, 'JPEGImages')
    annotations_folder = os.path.join(datadir, 'Annotations')

    # 获取所有图像文件名
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]
    
    for i, file in enumerate(image_files):
        with open(os.path.join(datadir, 'val.txt'), 'w') as train_file:
            train_file.write(file.split('.')[0] + '\n')

        val_transformer = build_transform(args, is_train=False)
        val_dataset = build_dataset(args, is_train=False, transformer=val_transformer)

        # VOC evaluation
        evaluator = build_eval(args, val_dataset, device)
        map, show_image = evaluator.eval(model)
        
        if map < threshold:
            dis_image = os.path.join('D:\SV_Person_Detection\data\TMP\JPEGImages', file)
            dis_label = os.path.join('D:\SV_Person_Detection\data\TMP\Annotations', file.replace('jpg', 'xml'))
            shutil.move(os.path.join(images_folder, file), dis_image)
            shutil.move(os.path.join(annotations_folder, file.replace('jpg', 'xml')), dis_label)
            # shutil.copy(os.path.join(images_folder, file), dis_image)
            # shutil.copy(os.path.join(annotations_folder, file.replace('jpg', 'xml')), dis_label)
            cv2.imwrite(os.path.join('D:\SV_Person_Detection\data\TMP\Visualization', file), show_image)
        print(f'{i}/{len(image_files)}')
