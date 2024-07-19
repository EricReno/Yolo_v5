import os
import torch
import numpy as np
import torch.nn as nn

from model.neck import SPPF
from model.head import DecoupleHead
from model.backbone import build_backbone

class YOLOv1(nn.Module):
    def __init__(self, 
                 args,
                 device,
                 trainable = False,
                 nms_thresh = 0.7,
                 conf_thresh = 0.001,
                 ):
        super(YOLOv1, self).__init__()

        self.deploy = False
        self.device = device                     
        self.num_classes = args.num_classes
        self.pretrained = args.pretrained
        self.pretrained_pth = os.path.join(args.root, args.project, 'results')

        self.conf_thresh = conf_thresh                 # 得分阈值
        self.nms_thresh = nms_thresh                   # NMS阈值
        self.stride = 32                               # 网络的最大步长
        self.trainable = trainable

        ## Backbone
        self.backbone, feat_dim = build_backbone(args.backbone, self.pretrained, self.pretrained_pth)

        ## Neck
        self.neck = SPPF(in_dim = feat_dim,
                         out_dim = feat_dim,
                         expand_ratio = args.expand_ratio,
                         pooling_size = args.pooling_size,
                         )

        ## Head
        self.head = DecoupleHead(in_dim = 512, 
                                 out_dim = 512, 
                                 num_classes = args.num_classes)

        ## Prediction
        self.obj_pred = nn.Conv2d(512, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(512, args.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(512, 4, kernel_size=1)

    def create_grid(self, fmp_size):
        """ 
            用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        """
        # 特征图的宽和高
        ws, hs = fmp_size

        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)],  indexing='ij')

        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        
        return grid_xy

    def decode_boxes(self, pred, fmp_size):
        """
            将txtytwth转换为常用的x1y1x2y2形式。
        """
        # 生成网格坐标矩阵
        grid_cell = self.create_grid(fmp_size)

        # 计算预测边界框的中心点坐标和宽高
        pred_ctr = (torch.sigmoid(pred[..., :2]) + grid_cell) * self.stride
        pred_wh = torch.exp(pred[..., 2:]) * self.stride

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

    ## basic NMS
    def nms(self, bboxes, scores, nms_thresh):
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
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(iou <= nms_thresh)[0]
            order = order[inds + 1]

        return keep
  
    ## class-aware NMS 
    def multiclass_nms_class_aware(self, scores, labels, bboxes, nms_thresh, num_classes):
        # nms
        keep = np.zeros(len(bboxes), dtype=np.int32)
        for i in range(num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, nms_thresh)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        return scores, labels, bboxes

    def postprocess(self, bboxes, scores):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]
        """

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        print(len(bboxes))
        print(bboxes)
        import time
        time.sleep(100)


        # nms
        scores, labels, bboxes = self.multiclass_nms_class_aware(
            scores, labels, bboxes, self.nms_thresh, self.num_classes)

        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        #测试阶段的前向推理代码
        
        # 主干网络
        feat = self.backbone(x)

        # 颈部网络
        feat = self.neck(feat)

        # 检测头
        cls_feat, reg_feat = self.head(feat)
        
        # 预测层
        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        
        fmp_size = obj_pred.shape[-2:]

        # 对 pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 测试时，笔者默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        obj_pred = obj_pred[0]       # [H*W, 1]
        cls_pred = cls_pred[0]       # [H*W, NC]
        reg_pred = reg_pred[0]       # [H*W, 4]

        # 每个边界框的得分
        scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())

        # 解算边界框, 并归一化边界框: [H*W, 4]
        bboxes = self.decode_boxes(reg_pred, fmp_size)

        if self.deploy:
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)
            return outputs
        else:
            scores = scores.cpu().numpy()
            bboxes = bboxes.cpu().numpy()

            # 后处理
            bboxes, scores, labels = self.postprocess(bboxes, scores)

        outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }

        return outputs

    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            # 主干网络
            feat = self.backbone(x)
    
            # 颈部网络
            feat = self.neck(feat)

            # 检测头
            cls_feat, reg_feat = self.head(feat)

            # 预测层
            obj_pred = self.obj_pred(cls_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            # 对 pred 的size做一些view调整，便于后续的处理
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # decode bbox
            box_pred = self.decode_boxes(reg_pred, fmp_size)

            # 网络输出
            outputs = {"pred_obj": obj_pred,                  # (Tensor) [B, M, 1]
                        "pred_cls": cls_pred,                 # (Tensor) [B, M, C]
                        "pred_box": box_pred,                 # (Tensor) [B, M, 4]
                        "stride": self.stride,                # (Int)
                        "fmp_size": fmp_size                  # (List) [fmp_h, fmp_w]
                        }           
            return outputs