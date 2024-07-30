import torch
import torch.nn as nn
from utils import Conv2d

class Reorg(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, ):
        super(Reorg, self).__init__()

        self.conv1 = Conv2d(512, 512*4)
        self.maxpool = nn.MaxPool2d(5, stride=2, padding=5//2)
        self.conv2 = Conv2d(512*4, 256)

        self.conv3 = Conv2d(1280, 1024)

    def forward(self, x1, x2):
        f1 = self.conv1(x1)
        f2 = self.maxpool(f1)
        f3 = self.conv2(f2)
        
        return self.conv3(torch.cat((f3, x2), 1))