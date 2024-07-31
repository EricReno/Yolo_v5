import torch.nn as nn
from .utils import Conv2d

class Decouple(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, ):
        super(Decouple, self).__init__()

        self.cls_feats = nn.Sequential(
            Conv2d(input_channles=1024, output_channles=512, kernel_size=3, stride=1, padding=1),
            Conv2d(input_channles=512, output_channles=512, kernel_size=3, stride=1, padding=1),
        )

        self.reg_feats = nn.Sequential(
            Conv2d(input_channles=1024, output_channles=512, kernel_size=3, stride=1, padding=1),
            Conv2d(input_channles=512, output_channles=512, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, x):
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)
       
        return cls_feats, reg_feats