import torch
import numpy as np
import torch.nn.functional as F

class Yolov1Loss(object):
    def __init__(self, 
                 device, 
                 num_classes,
                 boxes_per_cell,
                 loss_box_weight,
                 loss_noobj_weight
                 ):
        self.device = device
        self.num_classes = num_classes
        self.boxes_per_cell = boxes_per_cell
        self.loss_box_weight = loss_box_weight
        self.loss_noobj_weight = loss_noobj_weight

    def decode_gt(self, targets, stride, feature_size):
        batch_size = len(targets)
        gt_obj = np.zeros([batch_size, feature_size[0], feature_size[1], 1])
        gt_cls = np.zeros([batch_size, feature_size[0], feature_size[1], self.num_classes])
        gt_box = np.zeros([batch_size, feature_size[0], feature_size[1], 4])

        for index, target in enumerate(targets):
            target_cls = target['labels'].numpy()
            target_box = target['boxes'].numpy()

            for box, label in zip(target_box, target_cls):
                center_x, center_y = (box[0] + box[2])/2, (box[1] + box[3])/2, 

                grid_x = int(center_x / stride)
                grid_y = int(center_y / stride)

                gt_obj[index, grid_y, grid_x] = 1.0

                gt_cls[index, grid_y, grid_x] = np.zeros(self.num_classes)
                gt_cls[index, grid_y, grid_x][int(label)] = 1.0

                gt_box[index, grid_y, grid_x] = np.array([(box[0]+box[2])//2,
                                                          (box[1]+box[3])//2,
                                                          box[2] - box[0], 
                                                          box[3] - box[1]])
        gt_cls = np.expand_dims(gt_cls, axis=3)
        gt_cls = np.tile(gt_cls, (1, 1, 1, self.boxes_per_cell, 1)).reshape(-1, self.num_classes)
        gt_obj = np.expand_dims(gt_obj, axis=3)
        gt_obj = np.tile(gt_obj, (1, 1, 1, self.boxes_per_cell, 1)).reshape(-1)
        gt_box = np.expand_dims(gt_box, axis=3)
        gt_box = np.tile(gt_box, (1, 1, 1, self.boxes_per_cell, 1)).reshape(-1, 4)

        gt_cls = torch.from_numpy(gt_cls).to(self.device)    
        gt_obj = torch.from_numpy(gt_obj).to(self.device)
        gt_box = torch.from_numpy(gt_box).to(self.device)

        return gt_obj, gt_cls, gt_box

    def loss_objectness(self, pred_obj, gt_obj, pred_box, gt_box, stride, fmp_size):
        # regression loss
        gt_area = torch.mul(gt_box[:, 2], gt_box[:, 3])
        pred_area = torch.mul(pred_box[:, 2], pred_box[:, 3])

        w_intersect = \
            (torch.min(gt_box[:, 0] + gt_box[:, 2]/2, pred_box[:, 0] + pred_box[:, 2]/2).clamp(max=stride*fmp_size[0]) \
            - torch.max(gt_box[:, 0] - gt_box[:, 2]/2, pred_box[:, 0] - pred_box[:, 2]/2).clamp(min=0)).clamp_(min=0)
        h_intersect = \
            (torch.min(gt_box[:, 1] + gt_box[:, 3]/2, pred_box[:, 1] + pred_box[:, 3]/2).clamp(max=stride*fmp_size[1]) \
            - torch.max(gt_box[:, 1] - gt_box[:, 3]/2, pred_box[:, 1] - pred_box[:, 3]/2).clamp(min=0)).clamp_(min=0)

        area_intersect = w_intersect * h_intersect
        area_union = gt_area + pred_area - area_intersect
        ious = area_intersect / area_union.clamp(min=torch.finfo(torch.float32).eps)

        return torch.square(ious-gt_obj)
    
    def loss_no_objectness(self, pred_obj):
        return torch.square(pred_obj)
    
    def loss_classes(self, pred_cls, gt_cls):
        return torch.square(pred_cls - gt_cls)

    def loss_bboxes(self, pred_box, gt_box):
        gt_box[..., 2:] = torch.sqrt(gt_box[..., 2:])
        pred_box[..., 2:] = torch.sqrt(pred_box[..., 2:])
        return torch.square(pred_box - gt_box)

    def __call__(self, outputs, targets):
        gt_obj, gt_cls, gt_box = self.decode_gt(
                                                targets = targets,
                                                stride=outputs['stride'],
                                                feature_size = outputs['fmp_size']
                                                )

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_obj = outputs['pred_obj'].view(-1)                     # [BM,]
        pred_cls = outputs['pred_cls'].view(-1, self.num_classes)   # [BM, C]
        pred_box = outputs['pred_box'].view(-1, 4)                  # [BM, 4]

        mask = (gt_obj > 0)

        pred_noobj = pred_obj[~mask]
        # obj loss
        loss_obj = self.loss_objectness(pred_obj[mask], gt_obj[mask], pred_box[mask], gt_box[mask], outputs['stride'], outputs['fmp_size']).sum()/mask.sum()
        # cls loss
        loss_cls = self.loss_classes(pred_cls[mask], gt_cls[mask]).sum()/mask.sum()
        # box loss
        loss_box = self.loss_bboxes(pred_box[mask], gt_box[mask]).sum()/mask.sum()
        # no_obj loss
        loss_noobj = self.loss_no_objectness(pred_noobj).sum()/mask.sum()

        # total loss
        losses = self.loss_noobj_weight * loss_noobj + \
                 loss_obj +\
                 loss_cls + \
                 self.loss_box_weight * loss_box
                 
        return dict(
                loss_obj = loss_obj,
                loss_cls = loss_cls,
                loss_box = loss_box,
                loss_noobj = loss_noobj,
                losses = losses
        )