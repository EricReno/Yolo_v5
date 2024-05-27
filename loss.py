import torch
import numpy as np
import torch.nn.functional as F

class Criterion(object):
    def __init__(self, 
                 device, 
                 num_classes,
                 loss_obj_weight,
                 loss_cls_weight,
                 loss_box_weight
                 ):
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_box_weight = loss_box_weight

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
                center_w, center_h = box[2] - box[0], box[3] - box[1]

                grid_x = int(center_x / stride)
                grid_y = int(center_y / stride)

                gt_obj[index, grid_y, grid_x] = 1.0

                gt_cls[index, grid_y, grid_x] = np.zeros(self.num_classes)
                gt_cls[index, grid_y, grid_x][int(label)] = 1.0

                gt_box[index, grid_y, grid_x] = np.array([box[0], box[1], box[2], box[3]])

        gt_obj = gt_obj.reshape(-1)
        gt_cls = gt_cls.reshape(-1, self.num_classes)
        gt_box = gt_box.reshape(-1, 4)

        gt_obj = torch.from_numpy(gt_obj).to(self.device)
        gt_cls = torch.from_numpy(gt_cls).to(self.device)
        gt_box = torch.from_numpy(gt_box).to(self.device)

        return gt_obj, gt_cls, gt_box

    def loss_objectness(self, pred_obj, gt_obj):
        loss_obj = F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')

        return loss_obj
    
    def loss_classes(self, pred_cls, gt_label):
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_label, reduction='none')

        return loss_cls

    def loss_bboxes(self, pred_box, gt_box):
        # regression loss
        gt_area = (gt_box[..., 2] - gt_box[..., 0]) * (gt_box[..., 3] - gt_box[..., 1])
        pred_area = (pred_box[..., 2] - pred_box[..., 0]).clamp_(min=0) * (pred_box[..., 3] - pred_box[..., 1]).clamp_(min=0)

        w_intersect = (torch.min(gt_box[..., 2], pred_box[..., 2]) - torch.max(gt_box[..., 0], pred_box[..., 0])).clamp_(min=0)
        h_intersect = (torch.min(gt_box[..., 3], pred_box[..., 3]) - torch.max(gt_box[..., 1], pred_box[..., 1])).clamp_(min=0)

        area_intersect = w_intersect * h_intersect
        area_union = gt_area + pred_area - area_intersect
        ious = area_intersect / area_union.clamp(min=torch.finfo(torch.float32).eps)

        ## GIoU
        g_w_intersect = torch.max(gt_box[..., 2], pred_box[..., 2]) - torch.min(gt_box[..., 0], pred_box[..., 0])
        g_h_intersect = torch.max(gt_box[..., 3], pred_box[..., 3]) - torch.min(gt_box[..., 1], pred_box[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=torch.finfo(torch.float32).eps)
        
        return 1-gious

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

        loss_obj = self.loss_objectness(pred_obj, gt_obj).sum()/mask.sum()

        # cls loss
        pred_cls, gt_cls = pred_cls[mask], gt_cls[mask]
        loss_cls = self.loss_classes(pred_cls, gt_cls).sum()/mask.sum()
        
        # box loss
        pred_box, gt_box = pred_box[mask], gt_box[mask]
        loss_box = self.loss_bboxes(pred_box, gt_box).sum()/mask.sum()

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        loss_dict = dict(
                loss_obj = loss_obj,
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict