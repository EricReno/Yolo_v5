import torch.nn as nn
from .utils import Conv

class DecoupleHead(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 num_classes) -> None:
        super().__init__()

        # cls head
        cls_feats = [
            Conv(in_dim, out_dim, k=3, p=1, s=1),
            Conv(out_dim, out_dim, k=3, p=1, s=1)
        ]
        
        # reg head
        reg_feats = [
            Conv(in_dim, out_dim, k=3, p=1, s=1),
            Conv(out_dim, out_dim, k=3, p=1, s=1)
        ]
                     
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

    def forward(self, x):
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)
        
        return cls_feats, reg_feats