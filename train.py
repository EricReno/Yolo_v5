import time
import torch
torch.autograd.set_detect_anomaly(True)
import numpy
import argparse

# from eval import VOCEvaluator
from model.yolo_v1 import YOLOv1
from metric.criterion import Yolov1Loss
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

    parser.add_argument('--batch_size',     default=1,                     help='The batch size used by a single GPU during training')
    parser.add_argument('--image_size',     default=448,                    help='Input image size')
    parser.add_argument('--data_root',      default='E://data/VOCdevkit',   help='The path where the dataset is stored')
    parser.add_argument('--data_augment',   default=['RandomSaturationHue', 'RandomContrast', 'RandomBrightness', 'RandomSampleCrop', 'RandomExpand', 
                           'RandomHorizontalFlip'],                         help="[RandomExpand,RandomCenterCropPad,RandomCenterCropPad, RandomBrightness], default Resize")
    parser.add_argument('--datasets_val',   default=[('2007', 'test')],     help='The data set to be tested')
    parser.add_argument('--datasets_train', default=[('2007', 'trainval'), ('2012', 'trainval')], help='The data set to be trained')
    parser.add_argument('--class_names',    default= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
                           'sheep', 'sofa', 'train', 'tvmonitor'],          help='The category of predictions that the model can cover')
    parser.add_argument('--classes_number', default=20,                     help='The number of the classes')

    
    parser.add_argument('--epoch_max',      default=135,                    help='The maximum epoch')
    parser.add_argument('--epoch_warmup',   default=1,                      help='Epoch for warm_up')
    parser.add_argument('--epoch_second',   default=75,                     help='Epoch for second')
    parser.add_argument('--epoch_thirdly',  default=105,                    help='Epoch for finally')

    parser.add_argument('--lr',             default=0.01,                   help='Learning rate.')
    parser.add_argument('--lr_warmup',      default=0.001,                  help='Lr epoch for warm_up')
    parser.add_argument('--lr_second',      default=0.001,                  help='Lr epoch for second')
    parser.add_argument('--lr_thirdly',     default=0.0001,                 help='Lr epoch for finally')
    parser.add_argument('--lr_momentum',    default=0.9,                    help='Lr_momentum')
    parser.add_argument('--lr_weight_decay',default=0.0005,                 help='Lr_weight_decay')

    parser.add_argument('--boxes_per_cell', default=2,                      help='The number of the boxes in one cell')
    parser.add_argument('--loss_box_weight',default=5.0,                    help='The number of the classes')
    parser.add_argument('--loss_noobj_weight',default=0.5,                  help='The number of the classes')

    parser.add_argument('--threshold_nms',  default=0.5,                    help='NMS threshold')
    parser.add_argument('--threshold_conf', default=0.3,                    help='confidence threshold')
    # # parser.add_argument('--root', default='/data/ryb/', help='The root directory where code and data are stored')
    # parser.add_argument('--data', default='data/VOCdevkit', help='The path where the dataset is stored')
    # parser.add_argument('--project', default='ObjectDetection_VOC20', help='The path where the project code is stored')
    # parser.add_argument('--print_frequency', default=10, type=int, help='The print frequency')

    # # data & model
    # parser.add_argument('--num_workers', default=16, help='epoch for warm_up')
    # parser.add_argument('--img_size',   default=448, type=int, help='input image size')
    
    
    # parser.add_argument('--backbone', default='resnet18', help=['resnet18', 'resnet34'])
    # parser.add_argument('--expand_ratio', default=0.5, help='The expand_ratio')
    # parser.add_argument('--pooling_size', default=5, help='The pooling size setting')

    
    # """
    # Train configuration
    # """




    # parser.add_argument('--warmup_momentum', default=0.8, help='epoch for warm_up')
    # parser.add_argument('--grad_accumulate', default=1, type=int, help='gradient accumulation')
    # parser.add_argument('--resume', default='None', type=str, help=['None','44.pth'])
    # parser.add_argument('--save_folder', default='results', help='The path for wights')
    # parser.add_argument('--save_epoch', default=0, help='The epoch when the model parameters are saved')
    # parser.add_argument('--pretrained', default=False, help='Whether to use pre-training weights')
    
    # parser.add_argument('--ema', action='store_true', default=False, help='Model EMA')
    # """
    # Evaluate configuration
    # """
    
    # parser.add_argument('--recall_thr', default=101, help='The threshold for recall')
    # parser.add_argument('--eval_weight', default='84.pth', type=str, help="Trained state_dict file path")
    # parser.add_argument('--threshold', default=0.5, help='The iou threshold')
    # parser.add_argument('--real_time', default=True, help='whether to real-time display detection results')
    # parser.add_argument('--fuse_conv_bn', action='store_true', default=False, help='fuse Conv & BN')

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
    model = YOLOv1(device = device,
                   batch_size=args.batch_size,
                   image_size=args.image_size,
                   nms_thresh=args.threshold_nms,
                   num_classes=args.classes_number,
                   conf_thresh = args.threshold_conf,
                   boxes_per_cell=args.boxes_per_cell
                   ).to(device)
    model.trainable = True
          
    criterion = Yolov1Loss(device = device,
                         num_classes = args.classes_number,
                         boxes_per_cell = args.boxes_per_cell,
                         loss_box_weight = args.loss_box_weight,
                         loss_noobj_weight = args.loss_noobj_weight,
                         )
    
    grad_accumulate = max(1, round(64 / args.batch_size))
    learning_rate = (grad_accumulate*args.batch_size/64)*args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=args.lr_momentum, weight_decay=args.lr_weight_decay)
    

    # max_mAP = 0
    start_epoch = 0
    # if args.resume != "None":
    #     ckt_pth = os.path.join(args.root, args.project, 'results', args.resume)
    #     checkpoint = torch.load(ckt_pth, map_location='cpu')
    #     max_mAP = checkpoint['mAP']
    #     start_epoch = checkpoint['epoch'] + 1
    #     model.load_state_dict(checkpoint["model"])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # evaluator = VOCEvaluator(
    #     device=device,
    #     data_dir = os.path.join(args.root, args.data),
    #     dataset = val_dataset,
    #     image_sets = args.val_sets,
    #     ovthresh = args.threshold,  
    #     class_names = args.class_names,
    #     recall_thre = args.recall_thr,
    #     )

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
            loss_obj, loss_cls, loss_box, loss_noobj, losses = criterion(outputs=outputs, targets=targets).values()
            if grad_accumulate > 1:
               loss_obj /= grad_accumulate
               loss_cls /= grad_accumulate
               loss_box /= grad_accumulate
               loss_noobj /= grad_accumulate
               losses /= grad_accumulate
            losses.backward()
            
            # optimizer.step
            if ni % grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            ## log
            print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.4f}, Loss: {:8.4f}, Loss_obj: {:6.3f}, Loss_cls: {:6.3f}, Loss_box: {:6.3f}, Loss_noobj: {:6.3f}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epoch_max, iteration, len(train_dataloader), optimizer.param_groups[0]['lr'], losses, loss_obj, loss_cls, loss_box, loss_noobj))
            train_loss += losses.item() * images.size(0)
        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        # if epoch % 2 == 0:
        #     model = model if model_ema is None else model_ema.ema
        #     model.eval()
        #     val_loss = 0.0  
        #     with torch.no_grad():
        #         for iteration, (images, targets) in enumerate(val_dataloader):
        #             images = images.to(device).float()
        #             outputs = model(images)  
        #             loss_dic = criterion(outputs=outputs, targets=targets)
        #             losses = loss_dic['losses'] #[loss_obj, loss_cls, loss_box, losses]

        #             val_loss += losses.item() * images.size(0) 
        #     val_loss /= len(val_dataloader.dataset) 
        #     writer.add_scalar('Loss/val', val_loss, epoch)  

        #     # save_model
        #     if epoch >= args.save_epoch:
        #         model.trainable = False
        #         model.nms_thresh = args.nms_thresh
        #         model.conf_thresh = args.conf_thresh

        #         weight_name = '{}.pth'.format(epoch)
        #         result_path = os.path.join(args.root, args.project, args.save_folder, str(epoch))
        #         checkpoint_path = os.path.join(args.root, args.project, args.save_folder, weight_name)
                
        #         with torch.no_grad():
        #             mAP = evaluator.evaluate(model, result_path)
                
        #         writer.add_scalar('mAP', mAP, epoch)
        #         print("Epoch [{}]".format('-'*100))
        #         print("Epoch [{}:{}], mAP [{:.4f}]".format(epoch, args.max_epoch, mAP))
        #         print("Epoch [{}]".format('-'*100))
        #         if mAP > max_mAP:
        #             torch.save({'model': model.state_dict(),
        #                         'mAP': mAP,
        #                         'optimizer': optimizer.state_dict(),
        #                         'epoch': epoch,
        #                         'args': args},
        #                         checkpoint_path)
        #             max_mAP = mAP
                
        #         model.train()
        #         model.trainable = True
        
        # # LR Schedule
        # lr_scheduler.step()

if __name__ == "__main__":
    train()