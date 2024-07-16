# Hello, YoloV1--VOC20
- YOLO v1 论文传送门：https://arxiv.org/abs/1506.02640

## 数据集: VOC 
- **test**: (VOC2007, test) : 4952
- **train**: (VOC2007, trainval), (VOC2012, trainval) : 16553
- **CLASSES_NAMES**: 
|           |         |        |         |          | 
|   :---:   |  :---:  | :---:  |  :---:  | :---:    | 
| aeroplane | bicycle |  bird  |  boat   | bottle   | 
|    bus    |   car   |  cat   |  chair  | cow      | 
|diningtable|   dog   | horse  |motorbike| person   | 
|pottedplant|  sheep  |  sofa  | train   | tvmonitor| 

- **官方网址** 
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html


## 通用设置
| Backbone | Size  |  BS | Pretrain|Augment| Epoch| Obj_Weight | Cls_Weight | Box_Weight | NMS_Thre | Conf_Thre|
|  :---:   | :---: |:---:|  :---:  | :---: |:---: |   :---:    |:---:       | :---:      | :---:    | :---:    |
| Resnet18 |640x640|  32 |   CoCo  | None  | 150  |   1.0      | 1.0        | 5.0        | 0.5      |  0.3     |


## LearningRate: lr = (bs/64) * args.lr
| Name  | args.LR|  WarmUp |  OPT   |Momentum_Decay|             |   mAP    |       |
| :---: | :---:  |  :---:  | :---:  | :---:        |    :---:    |  :---:   | :---: |
|event_0| 0.0001 |  False  |  SGD   |   --         | Underfitting|17.04%(47)|       |
|event_1| 0.001  |  False  |  SGD   |   --         | Overfitting |28.80%(21)|       |
|event_2| 0.01   |  [0-1]  |  SGD   |   --         |      --     |00.00%(05)|       |
|event_3| 0.01   |  [0-3]  |  SGD   | 0.937_0.0005 | Overfitting |46.21%(04)| √     |

## DataAugment:
| Name  |    Augmentation   |   mAP    |         |
| :---: |        :---:      |  :---:   |  :---:  |
|event_4|+R_Bri_Sat_Hue_Filp|  52.40%  |         |
|event_5|  +R_Crop_Padding  |  67.47%  |         |

<img src="1.jpg">
