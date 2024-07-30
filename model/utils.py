import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, input_channles, output_channles, kernel_size = 1, stride = 1, padding = 0, dilation = 1) -> None:
        super(Conv2d, self).__init__()
        
        convs = []
        convs.append(nn.Conv2d(input_channles, output_channles, kernel_size, stride, padding, dilation, groups=1, bias=False))
        convs.append(nn.LeakyReLU(negative_slope=0.1))
        convs.append(torch.nn.BatchNorm2d(output_channles))
        
        self.convs = torch.nn.Sequential(*convs)
    
    def forward(self, x):
        return self.convs(x)