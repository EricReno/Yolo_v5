import os
import time
import torch
import numpy
import argparse

from model.yolo_v2 import Yolo_V2
from metric.criterion import V2_Loss
from dataset.voc import VOCDataset
from dataset.utils import CollateFunc
from dataset.augment import Augmentation
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection')

    """
    General configuration
    """
    parser.add_argument('--cuda',           default=True,                   help='Weather use cuda.')
    parser.add_argument('--worker_number',  default=1,                     help='Number Of Logical Processors')

    parser.add_argument('--batch_size',     default=8,                     help='The batch size used by a single GPU during training')
    parser.add_argument('--image_size',     default=416,                    help='Input image size')
    parser.add_argument('--data_root',      default='E://data/VOCdevkit',   help='The path where the dataset is stored')
    parser.add_argument('--data_augment',   default=['RandomSaturationHue', 'RandomContrast', 'RandomBrightness', 'RandomSampleCrop', 'RandomExpand', 
                           'RandomHorizontalFlip'],                         help="[RandomExpand,RandomCenterCropPad,RandomCenterCropPad, RandomBrightness], default Resize")
    parser.add_argument('--datasets_val',   default=[('2007', 'test')],     help='The data set to be tested')
    parser.add_argument('--datasets_train', default=[('2007', 'trainval'), ('2012', 'trainval')], help='The data set to be trained')
    parser.add_argument('--class_names',    default= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
                           'sheep', 'sofa', 'train', 'tvmonitor'],          help='The category of predictions that the model can cover')
    parser.add_argument('--classes_number', default=20,                     help='The number of the classes')

    
    parser.add_argument('--epoch_max',      default=160,                    help='The maximum epoch')
    parser.add_argument('--epoch_warmup',   default=1,                      help='Epoch for warm_up')
    parser.add_argument('--epoch_second',   default=60,                     help='Epoch for second')
    parser.add_argument('--epoch_thirdly',  default=90,                    help='Epoch for finally')

    parser.add_argument('--lr',             default=0.001,                   help='Learning rate.')
    parser.add_argument('--lr_warmup',      default=0,                  help='Lr epoch for warm_up')
    parser.add_argument('--lr_second',      default=0.0001,                  help='Lr epoch for second')
    parser.add_argument('--lr_thirdly',     default=0.00001,                 help='Lr epoch for finally')
    parser.add_argument('--lr_momentum',    default=0.9,                    help='Lr_momentum')
    parser.add_argument('--lr_weight_decay',default=0.0005,                 help='Lr_weight_decay')

    parser.add_argument('--boxes_per_cell', default=5,                      help='The number of the boxes in one cell')
    parser.add_argument('--loss_box_weight',default=5.0,                    help='The number of the classes')
    parser.add_argument('--loss_obj_weight',default=1.0,                    help='The number of the classes')
    parser.add_argument('--loss_cls_weight',default=1.0,                    help='The number of the classes')

    parser.add_argument('--threshold_nms',  default=0.5,                    help='NMS threshold')
    parser.add_argument('--threshold_conf', default=0.3,                    help='confidence threshold')

    return parser.parse_args()

def train():
    args = parse_args()
    writer = SummaryWriter('log')
    print("Setting Arguments.. : ")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("--------------------------------------------------------")

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ---------------------------- Build Datasets ----------------------------
    val_trans = Augmentation(args.image_size, args.data_augment, is_train=False)
    val_dataset = VOCDataset(data_dir     = args.data_root,
                             image_sets   = args.datasets_val,
                             transform    = val_trans,
                             is_train     = False)
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    val_b_sampler = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_b_sampler, collate_fn=CollateFunc(), num_workers=args.worker_number, pin_memory=True)
    
    train_trans = Augmentation(args.image_size, args.data_augment, is_train=True)
    train_dataset = VOCDataset(img_size   = args.image_size,
                               data_dir   = args.data_root,
                               image_sets = args.datasets_train,
                               transform  = train_trans,
                               is_train   = True)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_b_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_b_sampler, collate_fn=CollateFunc(), num_workers=args.worker_number, pin_memory=True)

    # ----------------------- Build Model ----------------------------------------
    model = Yolo_V2(device = device,
                   image_size=args.image_size,
                   nms_thresh=args.threshold_nms,
                   num_classes=args.classes_number,
                   conf_thresh = args.threshold_conf,
                   boxes_per_cell=args.boxes_per_cell
                   ).to(device)
    model.trainable = True
          
    criterion =  V2_Loss(device = device,
                         num_classes = args.classes_number,
                         boxes_per_cell = args.boxes_per_cell,
                         loss_box_weight = args.loss_box_weight,
                         loss_obj_weight = args.loss_obj_weight,
                         loss_cls_weight = args.loss_cls_weight,
                         )
    
    grad_accumulate = max(1, round(64 / args.batch_size))
    learning_rate = (grad_accumulate*args.batch_size/64)*args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=args.lr_momentum, weight_decay=args.lr_weight_decay)
    

    start_epoch = 0
    # ----------------------- Build Train ----------------------------------------
    start = time.time()
    for epoch in range(start_epoch, args.epoch_max):
        model.train()
        train_loss = 0.0
        for iteration, (images, targets) in enumerate(train_dataloader):
            ## learning rate
            ni = iteration + epoch * args.batch_size
            if epoch < args.epoch_warmup:
                optimizer.param_groups[0]['lr'] = numpy.interp(epoch*len(train_dataloader)+iteration,
                                                               [0, args.epoch_warmup*len(train_dataloader)],
                                                               [args.lr_warmup, learning_rate])
            elif epoch >= args.epoch_second:
                optimizer.param_groups[0]['lr'] = args.lr_second
            elif epoch >= args.epoch_thirdly:
                optimizer.param_groups[0]['lr'] = args.lr_thirdly

            ## forward
            images = images.to(device)
            outputs = model(images)

            ## loss
            loss_dict = criterion(outputs=outputs, targets=targets)
            [loss_obj, loss_cls, loss_box, losses] = loss_dict.values()
            if grad_accumulate > 1:
               losses /= grad_accumulate
            losses.backward()
            
            # optimizer.step
            if ni % grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            ## log
            print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.4f}, Loss: {:8.4f}, Loss_obj: {:6.3f}, Loss_cls: {:6.3f}, Loss_box: {:6.3f}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epoch_max, iteration, len(train_dataloader), optimizer.param_groups[0]['lr'], losses, loss_obj, loss_cls, loss_box))
            train_loss += losses.item() * images.size(0)
        
        weight = '{}.pth'.format(epoch)
        ckpt_path = os.path.join(os.getcwd(), 'log', weight)
        if not os.path.exists(os.path.dirname(ckpt_path)): 
            os.makedirs(os.path.dirname(ckpt_path))
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args},
                    ckpt_path)
        
        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

if __name__ == "__main__":
    train()