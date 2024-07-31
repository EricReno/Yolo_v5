import torch
import torch.nn as nn
from .utils import Conv2d

model_urls = {
    "darknet19": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet19.pth",
}

class Darknet19(nn.Module):
    def __init__(self, ):
        super(Darknet19, self).__init__()
        
        self.layer1 = nn.Sequential(
            Conv2d(input_channles=3, output_channles=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            Conv2d(input_channles=32, output_channles=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            Conv2d(input_channles=64, output_channles=128, kernel_size=3, stride=1, padding=1),
            Conv2d(input_channles=128, output_channles=64, kernel_size=1, stride=1),
            Conv2d(input_channles=64, output_channles=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            Conv2d(input_channles=128, output_channles=256, kernel_size=3, stride=1, padding=1),
            Conv2d(input_channles=256, output_channles=128, kernel_size=1, stride=1),
            Conv2d(input_channles=128, output_channles=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            Conv2d(input_channles=256, output_channles=512, kernel_size=3, stride=1, padding=1),
            Conv2d(input_channles=512, output_channles=256, kernel_size=1, stride=1),
            Conv2d(input_channles=256, output_channles=512, kernel_size=3, stride=1, padding=1),
            Conv2d(input_channles=512, output_channles=256, kernel_size=1, stride=1),
            Conv2d(input_channles=256, output_channles=512, kernel_size=3, stride=1, padding=1),
        )
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer6 = nn.Sequential(
            Conv2d(input_channles=512, output_channles=1024, kernel_size=3, stride=1, padding=1),
            Conv2d(input_channles=1024, output_channles=512, kernel_size=1, stride=1),
            Conv2d(input_channles=512, output_channles=1024, kernel_size=3, stride=1, padding=1),
            Conv2d(input_channles=1024, output_channles=512, kernel_size=1, stride=1),
            Conv2d(input_channles=512, output_channles=1024, kernel_size=3, stride=1, padding=1),
        )

        self.layer7 = nn.Sequential(
            Conv2d(input_channles=1024, output_channles=1000, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=7),
            nn.Softmax()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        f6 = self.layer6(self.maxpool_5(f5))
        # f7 = self.layer7(f6)

        return f5, f6
    
if __name__ == "__main__":
    input = torch.randn(1, 3, 416, 416)

    model = Darknet19()

    output = model(input)

    print(output.shape)