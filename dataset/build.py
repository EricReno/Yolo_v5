import torch
from .voc import VOCDataset
from .augment.ssd_augment import SSDAugmentation
from .augment.yolo_augment import YOLOAugmentation

class CollateFunc(object):
    def __call__(self, batch):
        images = []
        targets = []

        for sample in batch:
            image = sample[0]
            target = sample[1]

            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, 0)

        return images, targets
    
def build_dataset(args, is_train, transformer):
    if is_train :
        print('==============================')
        print('Build Dataset: VOC ...')
        print('Dataset Class_names: {}'.format(args.class_names))
        datasets = VOCDataset(img_size       = args.image_size,
                              is_train       = True,
                              data_dir       = args.data_root,
                              transform      = transformer,
                              image_set      = args.train_dataset,
                              voc_classes    = args.class_names,
                              mosaic_augment = args.mosaic,
                              mixup_augment = args.mix_up
                              )
    else:
        datasets = VOCDataset(img_size       = args.image_size,
                              is_train       = False,
                              data_dir       = args.data_root,
                              transform      = transformer,
                              image_set      = args.val_dataset,
                              voc_classes    = args.class_names,
                              )
    return datasets
    
def build_transform(args, is_train):
    if args.data_augment == 'ssd':
        transform = SSDAugmentation(is_train=is_train, image_size=args.image_size)
    elif args.data_augment == 'yolo':
        transform = YOLOAugmentation(is_train=is_train, image_size=args.image_size, max_stride=32)

    return transform

def build_dataloader(args, dataset):
    sampler = torch.utils.data.RandomSampler(dataset)
    b_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)

    return dataloader