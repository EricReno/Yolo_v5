import torch
import argparse
import numpy as np
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, c1, c2, kernel_size = 1, stride = 1, padding = 0, dilation = 1) -> None:
        super(Conv2d, self).__init__()
        
        convs = []
        convs.append(nn.Conv2d(c1, c2, kernel_size, stride, padding, dilation, groups=1, bias=False))
        convs.append(nn.LeakyReLU(negative_slope=0.1))
        
        self.convs = torch.nn.Sequential(*convs)
    
    def forward(self, x):
        return self.convs(x)

class YOLOv1(nn.Module):
    def __init__(self, 
                 device,
                 batch_size,
                 image_size,
                 nms_thresh,
                 num_classes,
                 conf_thresh,
                 boxes_per_cell
                 ):
        super(YOLOv1, self).__init__()

        self.stride = 64                           
        self.deploy = False
        self.device = device
        self.trainable = False
        self.batch_size = batch_size
        self.image_size = image_size
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.boxes_per_cell = boxes_per_cell

        self.cell_size = self.image_size//self.stride
        self.boundary1 = self.cell_size * self.cell_size * self.num_classes                      #类似于 7*7*20
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell  #类似于 7*7*20 + 7*7*2
        
        layers1 = [
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        self.layer1 = nn.Sequential(*layers1)

        layers2 = [
            Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        self.layer2 = nn.Sequential(*layers2)

        layers3 = [
            Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        self.layer3 = nn.Sequential(*layers3)

        layers4 = []
        for _ in range(4):
            layers4.append(Conv2d(512, 256, kernel_size=1, stride=1, padding=0))
            layers4.append(Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        layers4.extend([
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)])
        self.layer4 = nn.Sequential(*layers4)

        layers5 = []
        for _ in range(2):
            layers5.append(Conv2d(1024, 512, kernel_size=1, stride=1, padding=0))
            layers5.append(Conv2d(512, 1024, kernel_size=3, stride=1, padding=1))
        layers5.extend([
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)])
        self.layer5 = nn.Sequential(*layers5)

        layers6 = [
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        ]
        self.layer6 = nn.Sequential(*layers6)

        layers7 = [
            nn.Linear(self.cell_size*self.cell_size*1024, 2048, bias=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, self.cell_size*self.cell_size*(self.boxes_per_cell*5+self.num_classes), bias=True),
        ]
        self.layer7 = nn.Sequential(*layers7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

        # [H, W, 2] -> [H, W, 2, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = torch.unsqueeze(grid_xy, dim=2)
        grid_xy = grid_xy.repeat(1, 1, self.boxes_per_cell, 1).view(-1, 2).to(self.device)

        return grid_xy

    def decode_boxes(self, pred, fmp_size):
        """
            将txtytwth转换为常用的x1y1x2y2形式。
        """
        grid_cell = self.create_grid(fmp_size)
        pred[..., :2] = (pred[..., :2] + grid_cell) * self.stride
        pred[..., 2:] = pred[..., 2:] * self.stride * fmp_size[0]

        return pred

    ## basic NMS
    def nms(self, bboxes, scores, nms_thresh):
        """"Pure Python NMS."""
        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        pred_x1y1 = bboxes[..., :2] - bboxes[..., 2:] * 0.5
        pred_x2y2 = bboxes[..., :2] + bboxes[..., 2:] * 0.5
        bboxes = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)

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

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int32)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_thresh)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]
        
        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, cls_pred, obj_pred, reg_pred, fmp_size):
        # 测试时，笔者默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        obj_pred = obj_pred[0]       # [H*W, 1]
        cls_pred = cls_pred[0]       # [H*W, C]
        reg_pred = reg_pred[0]       # [C, H*W, 4]

        # 每个边界框的得分
        scores = obj_pred.sigmoid()

        # 解算边界框, 并归一化边界框: [H*W, 4]
        bboxes = self.decode_boxes(reg_pred, fmp_size)

        if self.deploy:
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)
            return outputs
        else:
            scores = scores.numpy()
            bboxes = bboxes.numpy()

            # 后处理
            bboxes, scores, labels = self.postprocess(bboxes, scores)

        outputs = {
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }

        return outputs

    def forward(self, x):
        f1 = self.layer1(x)    #[B, 192, H/4, W/4]
        f2 = self.layer2(f1)   #[B, 256, H/8, W/8]
        f3 = self.layer3(f2)   #[B, 512, H/16, W/16]
        f4 = self.layer4(f3)   #[B, 1024, H/32, W/32]
        f5 = self.layer5(f4)   #[B, 1024, H/64, W/64]
        f6 = self.layer6(f5)   #[B, 1024, H/64, W/64]
        f7 = self.layer7(f6.flatten(1, 3))  #[B, 30*H/64*W/64]
        f7_clamped = torch.clamp(f7, min=-5, max=5)
        pred = torch.sigmoid(f7_clamped)

        # 对 pred 的size做一些view调整，便于后续的处理
        # -> [B, H*W*BPC, 20] 
        # -> [B, H*W*BPC, 1] 
        # -> [B, H*W*BPC, 4] 
        cls_pred = pred[:,              :self.boundary1].reshape(self.batch_size, self.cell_size*self.cell_size, self.num_classes)
        cls_pred = torch.unsqueeze(cls_pred, dim=2)
        cls_pred = cls_pred.repeat(1, 1, self.boxes_per_cell, 1).reshape(self.batch_size, self.cell_size*self.cell_size*self.boxes_per_cell, self.num_classes).contiguous()                                                              
        obj_pred = pred[:,self.boundary1:self.boundary2].reshape(self.batch_size, self.cell_size*self.cell_size*self.boxes_per_cell, 1).contiguous()
        reg_pred = pred[:,self.boundary2:              ].reshape(self.batch_size, self.cell_size*self.cell_size*self.boxes_per_cell, 4).contiguous()
        
        fmp_size = [self.cell_size, self.cell_size]
        if not self.trainable:
            return self.inference(cls_pred, obj_pred, reg_pred, fmp_size)
        else:
            box_pred = self.decode_boxes(reg_pred, fmp_size)

            # 网络输出
            outputs = {"pred_obj": obj_pred,                 # (Tensor) [B, M, 1]
                       "pred_cls": cls_pred,                 # (Tensor) [B, M, C]
                       "pred_box": box_pred,                 # (Tensor) [B, M, 4]
                       "stride": self.stride,                # (Int)
                       "fmp_size": fmp_size                  # (List) [fmp_h, fmp_w]
                       }
            return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yolo v1')
    parser.add_argument('--cuda',           default=False,  help='Weather use cuda.')
    parser.add_argument('--batch_size',     default=1,      help='The batch size used by a single GPU during training')
    parser.add_argument('--image_size',     default=448,    help='input image size')
    parser.add_argument('--num_classes',    default=20,     help='The number of the classes')
    parser.add_argument('--boxes_per_cell', default=2,      help='The number of the boxes in one cell')
    parser.add_argument('--conf_thresh',    default=0.3,    help='confidence threshold')
    parser.add_argument('--nms_thresh',     default=0.5,    help='NMS threshold')

    args = parser.parse_args()
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input = torch.randn(1, 3, 448, 448)

    model = YOLOv1(
        device = device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        nms_thresh=args.nms_thresh,
        num_classes=args.num_classes,
        conf_thresh = args.conf_thresh,
        boxes_per_cell=args.boxes_per_cell
        )
    model.trainable = False

    output = model(input)
    print(output)