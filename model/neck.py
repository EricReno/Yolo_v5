import torch
import torch.nn as nn
from .utils import Conv

class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio, pooling_size) -> None:
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)

        self.cv1 = Conv(in_dim, inter_dim, k = 1)

        self.m = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

        self.cv2 = Conv(inter_dim*4, out_dim, k = 1)

    def forward(self, x):
        x = self.cv1(x)  ## 1
        y1 = self.m(x)   ## 5
        y2 = self.m(y1)  ## 9
        y3 = self.m(y2)  ## 13

        return self.cv2(torch.cat((x, y1, y2, y3), 1))