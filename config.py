import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection')

    """
    General configuration
    """
    # path
    parser.add_argument('--root', default='/data/ryb/', help='The root directory where code and data are stored')
    parser.add_argument('--data', default='data/VOCdevkit', help='The path where the dataset is stored')
    parser.add_argument('--project', default='Yolo', help='The path where the project code is stored')
    # env
    parser.add_argument('--cuda', default=True, help='Weather use cuda.')
    parser.add_argument('--print_frequency', default=10, type=int, help='The print frequency')

    # data
    parser.add_argument('--img_size',   default=640, type=int, help='input image size')
    parser.add_argument('--val_sets',   default=[('2007', 'test')], help='The data set to be tested')
    parser.add_argument('--train_sets', default=[('2007', 'trainval'), ('2012', 'trainval')], help='The data set to be trained')
    parser.add_argument('--num_classes', default=20, help='The number of the classes')
    parser.add_argument('--class_names', default= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                                                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
                                         help= 'The category of predictions that the model can cover')
    # model yolov1
    parser.add_argument('--backbone', default='resnet18', help=['resnet18', 'resnet34'])
    parser.add_argument('--expand_ratio', default=0.5, help='The expand_ratio')
    parser.add_argument('--pooling_size', default=5, help='The pooling size setting')
    parser.add_argument('--loss_obj_weight', default=1.0, help='The number of the classes')
    parser.add_argument('--loss_cls_weight', default=1.0, help='The number of the classes')
    parser.add_argument('--loss_box_weight', default=5.0, help='The number of the classes')
    
    """
    Train configuration
    """
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate.')
    parser.add_argument('--lr_momentum', default=0.937, help='lr_momentum')
    parser.add_argument('--lr_weight_decay', default=0.0005, help='lr_weight_decay')
    parser.add_argument('--warmup_epoch', default=5, help='epoch for warm_up')

    parser.add_argument('--resume', default='None', type=str, help=['None','44.pth'])
    parser.add_argument('--batch_size', default=32, help='The batch size used by a single GPU during training')
    parser.add_argument('--save_folder', default='results', help='The path for wights')
    parser.add_argument('--max_epoch', default=150, help='The maximum epoch used in this training')
    parser.add_argument('--save_epoch', default=0 , help='The epoch when the model parameters are saved')
    parser.add_argument('--pretrained', default=True, help='Whether to use pre-training weights')
    parser.add_argument('--data_augmentation', 
                        default=[
                            'RandomSaturationHue',
                            'RandomContrast',
                            'RandomBrightness',
                            'RandomSampleCrop',
                            'RandomExpand',
                            'RandomHorizontalFlip',
                            ],
                        help="[RandomExpand,RandomCenterCropPad,RandomCenterCropPad, RandomBrightness], default Resize")

    """
    Evaluate configuration
    """
    parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold')
    parser.add_argument('--conf_thresh', default=0.3, type=float, help='confidence threshold')
    parser.add_argument('--recall_thr', default=101, help='The threshold for recall')
    parser.add_argument('--weight', default='128.pth', type=str, help="Trained state_dict file path")
    parser.add_argument('--threshold', default=0.5, help='The iou threshold')
    parser.add_argument('--real_time', default=False, help='whether to real-time display detection results')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False, help='fuse Conv & BN')

    return parser.parse_args()