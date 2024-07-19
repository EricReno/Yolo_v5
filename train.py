import os
import time
import torch
import numpy

from loss import Criterion
from eval import VOCEvaluator
from config import parse_args
from model.yolov1 import YOLOv1
from model.utils import ModelEMA
from dataset.voc import VOCDataset
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
    val_trans = Augmentation(args.img_size, args.data_augmentation, is_train=False)
    val_dataset = VOCDataset(data_dir     = os.path.join(args.root, args.data),
                             image_sets   = args.val_sets,
                             transform    = val_trans,
                             is_train     = False)
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    val_b_sampler = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)
    
    train_trans = Augmentation(args.img_size, args.data_augmentation, is_train=True)
    train_dataset = VOCDataset(img_size   = args.img_size,
                               data_dir   = os.path.join(args.root, args.data),
                               image_sets = args.train_sets,
                               transform  = train_trans,
                               is_train   = True)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_b_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)

    # ----------------------- Build Model ----------------------------------------
    model = YOLOv1(args = args,
                   device = device,
                   trainable = True,
                   ).to(device)
      
    criterion = Criterion(device = device,
                          num_classes = args.num_classes,
                          loss_obj_weight = args.loss_obj_weight,
                          loss_cls_weight = args.loss_cls_weight,
                          loss_box_weight = args.loss_box_weight
                          )
    
    grad_accumulate = max(args.grad_accumulate, round(64 / args.batch_size))
    lr1 = (grad_accumulate*args.batch_size/64)*args.lr
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr1, momentum=args.lr_momentum, weight_decay=args.lr_weight_decay)
    

    max_mAP = 0
    start_epoch = 0
    if args.resume != "None":
        ckt_pth = os.path.join(args.root, args.project, 'results', args.resume)
        checkpoint = torch.load(ckt_pth, map_location='cpu')
        max_mAP = checkpoint['mAP']
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])  

    if args.ema:
        model_ema = ModelEMA({'ema_decay': 0.9998, 'ema_tau': 2000}, model, start_epoch * len(train_dataloader))
    else:
        model_ema = None

    evaluator = VOCEvaluator(
        device=device,
        data_dir = os.path.join(args.root, args.data),
        dataset = val_dataset,
        image_sets = args.val_sets,
        ovthresh = args.threshold,     
        class_names = args.class_names,
        recall_thre = args.recall_thr,
        )

    # ----------------------- Build Train ----------------------------------------
    start = time.time()
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        train_loss = 0.0
        for iteration, (images, targets) in enumerate(train_dataloader):
            ni = iteration + epoch * args.batch_size
            if epoch < args.warmup_epoch:
                optimizer.param_groups[0]['lr'] = numpy.interp(epoch*len(train_dataloader)+iteration+1,
                                                               [0, args.warmup_epoch*len(train_dataloader)],
                                                               [0, lr1])
                optimizer.param_groups[0]['momentum'] = numpy.interp(epoch*len(train_dataloader)+iteration+1,
                                                                     [0, args.warmup_epoch*len(train_dataloader)],
                                                                     [args.warmup_momentum, args.lr_momentum])

            images = images.to(device)

            # Inference
            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(images)

            # Compute loss
            loss_dic = criterion(outputs=outputs, targets=targets)
            loss_obj, loss_cls, loss_box, losses = loss_dic.values()
            if grad_accumulate > 1:
               loss_obj /= grad_accumulate
               loss_cls /= grad_accumulate
               loss_box /= grad_accumulate
               losses /= grad_accumulate
            
            # Backward
            losses.backward()
            
            if ni % grad_accumulate == 0:
                # optimizer.step
                optimizer.step()
                optimizer.zero_grad()

                # ema
                if model_ema is not None:
                    model_ema.update(model)

            if iteration % args.print_frequency == 0 or iteration == len(train_dataloader):
                print("Epoch [{}:{}/{}:{}], Time [{}] lr: {:4f}, Loss: {:.4f}, Loss_obj: {:.4f}, Loss_cls: {:.4f}, Loss_box: {:.4f}".
                    format(epoch, args.max_epoch, iteration, len(train_dataloader), time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                           optimizer.param_groups[0]['lr'], losses, loss_dic['loss_obj'], loss_dic['loss_cls'], loss_dic['loss_box']))
            
            train_loss += losses.item() * images.size(0)
        
        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        if epoch % 2 == 0:
            model = model if model_ema is None else model_ema.ema
            model.eval()
            val_loss = 0.0  
            with torch.no_grad():
                for iteration, (images, targets) in enumerate(val_dataloader):
                    images = images.to(device).float()
                    outputs = model(images)  
                    loss_dic = criterion(outputs=outputs, targets=targets)
                    losses = loss_dic['losses'] #[loss_obj, loss_cls, loss_box, losses]

                    val_loss += losses.item() * images.size(0) 
            val_loss /= len(val_dataloader.dataset) 
            writer.add_scalar('Loss/val', val_loss, epoch)  

            # save_model
            if epoch >= args.save_epoch:
                model.trainable = False
                model.nms_thresh = args.nms_thresh
                model.conf_thresh = args.conf_thresh

                weight_name = '{}.pth'.format(epoch)
                result_path = os.path.join(args.root, args.project, args.save_folder, str(epoch))
                checkpoint_path = os.path.join(args.root, args.project, args.save_folder, weight_name)
                
                with torch.no_grad():
                    mAP = evaluator.evaluate(model, result_path)
                
                writer.add_scalar('mAP', mAP, epoch)
                print("Epoch [{}]".format('-'*100))
                print("Epoch [{}:{}], mAP [{:.4f}]".format(epoch, args.max_epoch, mAP))
                print("Epoch [{}]".format('-'*100))
                if mAP > max_mAP:
                    torch.save({'model': model.state_dict(),
                                'mAP': mAP,
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch,
                                'args': args},
                                checkpoint_path)
                    max_mAP = mAP
                
                model.train()
                model.trainable = True

if __name__ == "__main__":
    train()