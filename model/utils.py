import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1, c2, k = 1, p = 0, s = 1, d = 1, act_type = 'lrelu', norm_type = 'BN') -> None:
        super(Conv, self).__init__()
        
        convs = []
        convs.append(nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=1, bias=False))
        convs.append(nn.LeakyReLU(c2))
        convs.append(nn.BatchNorm2d(c2))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)