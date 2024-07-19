import math
import torch
from copy import deepcopy

class Conv(torch.nn.Module):
    def __init__(self, c1, c2, k = 1, p = 0, s = 1, d = 1, act_type = 'lrelu', norm_type = 'BN') -> None:
        super(Conv, self).__init__()
        
        convs = []
        convs.append(torch.nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=1, bias=False))
        convs.append(torch.nn.LeakyReLU(c2))
        convs.append(torch.nn.BatchNorm2d(c2))

        self.convs = torch.nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)
    
class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, updates=0) -> None:
        self.ema = deepcopy(model).eval()
        # self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.decay = lambda x:decay * (1 - math.exp(-x /2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def is_parallel(self, model):
        # Returns True if model is of type DP or DDP
        return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)

    def de_parallel(self, model):
        # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
        return model.module if self.is_parallel(model) else model

    def copy_attr(self, a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def update(self, model): 
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
        
        msd = model.state_dict()  
        # msd = model.module.state_dict() if is_parallel(model) else model.state_dict() 
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.ema, model, include, exclude)
    
    