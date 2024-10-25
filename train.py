<<<<<<< HEAD
import os
import time
import torch
import numpy
from torch.utils.tensorboard import SummaryWriter

from config import parse_args
from evaluate import build_eval
from model.build import build_yolo
from utils.ema import ModelEMA
from utils.loss import build_loss
from utils.flops import compute_flops
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lambda_lr_scheduler
from utils.rescale import refine_targets, rescale_image_targets
from dataset.build import build_transform, build_dataset, build_dataloader

def train():
    args = parse_args()
    writer = SummaryWriter('deploy')
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ---------------------------- Build --------------------------
    val_transformer = build_transform(args, is_train=False)
    train_transformer = build_transform(args, is_train=True)

    val_dataset = build_dataset(args, is_train=False, transformer=val_transformer)
    train_dataset = build_dataset(args, True, train_transformer, )
    train_dataloader = build_dataloader(args, train_dataset)

    model = build_yolo(args, device, True)
    compute_flops(model, args.image_size, device)
          
    loss_function =  build_loss(args, device)
    
    evaluator = build_eval(args, val_dataset, device)
    
    optimizer, start_epoch = build_optimizer(args, model)

    lr_scheduler, lf = build_lambda_lr_scheduler(args, optimizer)
    if args.resume_weight_path and args.resume_weight_path != 'None':
        lr_scheduler.last_epoch = start_epoch - 1
        optimizer.step()
        lr_scheduler.step()
    
    if args.ema:
        model_ema = ModelEMA(model, start_epoch * len(train_dataloader))
    else:
        model_ema = None

    # ----------------------- Train --------------------------------
    print('==============================')
    max_mAP = 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs_total+1):
        model.train()
        train_loss = 0.0
        model.trainable = True
        for iteration, (images, targets) in enumerate(train_dataloader):
            ## learning rate
            ni = iteration + epoch * len(train_dataloader)
            if epoch < args.warmup_epochs:
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = numpy.interp(epoch*len(train_dataloader)+iteration,
                                           [0, args.warmup_epochs*len(train_dataloader)],
                                           [0.1 if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
            
            images = images.to(device)
             # Multi scale
            if args.multi_scale:
                images, targets, img_size = rescale_image_targets(
                    images, targets, [8, 16, 32], args.min_box_size, args.multi_scale)
            else:
                targets = refine_targets(targets, args.min_box_size)

            ## forward
            outputs = model(images)

            ## loss
            loss_dict = loss_function(outputs=outputs, targets=targets)
            [loss_obj, loss_cls, loss_box, losses] = loss_dict.values()
            if args.grad_accumulate > 1:
               losses /= args.grad_accumulate
               loss_obj /= args.grad_accumulate
               loss_cls /= args.grad_accumulate
               loss_box /= args.grad_accumulate
            losses.backward()
            
            # optimizer.step
            if ni % args.grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                # ema
                if model_ema is not None:
                   model_ema.update(model)

            print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.5f}, Loss: {:8.4f}, Loss_obj: {:8.4f}, Loss_cls: {:6.3f}, Loss_box: {:6.3f}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epochs_total, iteration+1, len(train_dataloader), optimizer.param_groups[2]['lr'], losses, loss_obj, loss_cls, loss_box))
            train_loss += losses.item() * images.size(0)

        lr_scheduler.step()

        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        # chech model
        model_eval = model if model_ema is None else model_ema.ema
        model_eval.eval()
        model_eval.trainable = False
        # save_model
        if epoch >= args.save_checkpoint_epoch:
            ckpt_path = os.path.join(os.getcwd(), 'deploy', 'best.pth')
            if not os.path.exists(os.path.dirname(ckpt_path)):
                os.makedirs(os.path.dirname(ckpt_path))
            
            with torch.no_grad():
                mAP = evaluator.eval(model_eval)
            writer.add_scalar('mAP', mAP, epoch)

            if mAP > max_mAP:
                torch.save({
                        'model': model_eval.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'mAP':mAP,
                        'epoch': epoch,
                        'args': args},
                        ckpt_path)
                max_mAP = mAP
        
if __name__ == "__main__":
=======
import os
import time
import torch
import numpy
from config import parse_args
from eval import Evaluator
from model.yolov3 import YOLOv3
from metric.criterion import Loss
from dataset.face import FACEDataset
from dataset.utils import CollateFunc
from dataset.augment import Augmentation
from torch.utils.tensorboard import SummaryWriter

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
    val_dataset = FACEDataset(data_dir     = args.data_root,
                             image_sets   = args.datasets_val,
                             transform    = val_trans,
                             is_train     = False,
                             classnames = args.class_names
                             )
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    val_b_sampler = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_b_sampler, collate_fn=CollateFunc(), num_workers=args.worker_number, pin_memory=True)
    
    train_trans = Augmentation(args.image_size, args.data_augment, is_train=True)
    train_dataset = FACEDataset(img_size   = args.image_size,
                               data_dir   = args.data_root,
                               image_sets = args.datasets_train,
                               transform  = train_trans,
                               is_train   = True,
                               classnames = args.class_names)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_b_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_b_sampler, collate_fn=CollateFunc(), num_workers=args.worker_number, pin_memory=True)

    # ----------------------- Build Model ----------------------------------------
    model = YOLOv3(device = device,
                   backbone = args.backbone,
                   image_size=args.image_size,
                   nms_thresh=args.threshold_nms,
                   anchor_size = args.anchor_size,
                   num_classes=args.classes_number,
                   conf_thresh = args.threshold_conf,
                   boxes_per_cell=args.boxes_per_cell
                   ).to(device)
          
    criterion =  Loss(device = device,
                         anchor_size = args.anchor_size,
                         num_classes = args.classes_number,
                         boxes_per_cell = args.boxes_per_cell,
                         loss_box_weight = args.loss_box_weight,
                         loss_obj_weight = args.loss_obj_weight,
                         loss_cls_weight = args.loss_cls_weight,
                         )
    
    evaluator = Evaluator(
        device   =device,
        data_dir = args.data_root,
        dataset  = val_dataset,
        image_sets = args.datasets_val,
        ovthresh    = args.threshold_nms,                        
        class_names = args.class_names,
        recall_thre = args.threshold_recall,
        )
    
    grad_accumulate = max(1, round(64 / args.batch_size))
    learning_rate = (grad_accumulate*args.batch_size/64)*args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=args.lr_momentum, weight_decay=args.lr_weight_decay)
    
    max_mAP = 0
    start_epoch = 0
    if args.weight != "None":
        ckt_pth = os.path.join(os.getcwd(), 'log', args.weight)
        checkpoint = torch.load(ckt_pth, map_location='cpu', weights_only=False)
        # max_mAP = checkpoint['mAP']
        # resume_epoch = checkpoint['epoch'] + 1         
        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint['optimizer'])

    # ----------------------- Build Train ----------------------------------------
    start = time.time()
    for epoch in range(start_epoch, args.epoch_max):
        model.train()
        train_loss = 0.0
        model.trainable = True
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
               loss_obj /= grad_accumulate
               loss_cls /= grad_accumulate
               loss_box /= grad_accumulate
            losses.backward()
            
            # optimizer.step
            if ni % grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            ## log
            print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.5f}, Loss: {:8.4f}, Loss_obj: {:8.4f}, Loss_cls: {:6.3f}, Loss_box: {:6.3f}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epoch_max, iteration+1, len(train_dataloader), optimizer.param_groups[0]['lr'], losses, loss_obj, loss_cls, loss_box))
            train_loss += losses.item() * images.size(0)
        
        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        val_loss = 0.0  
        with torch.no_grad():
            for iteration, (images, targets) in enumerate(val_dataloader):
                images = images.to(device)
                outputs = model(images)
                loss_dict = criterion(outputs=outputs, targets=targets)
                losses = loss_dict['losses'] #[loss_obj, loss_cls, loss_box, losses]
                print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.5f}, Loss: {:8.4f} ".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epoch_max, iteration+1, len(val_dataloader), optimizer.param_groups[0]['lr'], losses))
                val_loss += losses.item() * images.size(0) 
            val_loss /= len(val_dataloader.dataset) 
            writer.add_scalar('Loss/val', val_loss, epoch)
        
        # save_model
        if epoch >= args.epoch_save:
            model.trainable = False
            model.nms_thresh = args.threshold_nms
            model.conf_thresh = args.threshold_conf

            weight = '{}.pth'.format(epoch)
            ckpt_path = os.path.join(os.getcwd(), 'log', weight)
            if not os.path.exists(os.path.dirname(ckpt_path)): 
                os.makedirs(os.path.dirname(ckpt_path))
            
            with torch.no_grad():
                mAP = evaluator.eval(model)
            writer.add_scalar('mAP', mAP, epoch)
 
            if mAP > max_mAP:
                torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'mAP':mAP,
                        'epoch': epoch,
                        'args': args},
                        ckpt_path)
                max_mAP = mAP

if __name__ == "__main__":
>>>>>>> dd965f49a1e0ea3f477e008e2992bdd99cc3e8cc
    train()