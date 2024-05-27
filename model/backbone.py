import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out  

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        pass

    def forward(self, x):
        pass

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  ## 地址传递
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    pass
                    # nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input:
            x -> [B, C, H, W]
        OutPut:
            c5 -> [B, C, H/32, H/32]
        """
        c1 = self.conv1(x)   #[B, C, H/2, W/2]
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.maxpool(c1)#[B, C, H/4, W/4]

        c2 = self.layer1(c2) #[B, C, H/4, W/4]
        c3 = self.layer2(c2) #[B, C, H/8, W/8]
        c4 = self.layer3(c3) #[B, C, H/16, W/16]
        c5 = self.layer4(c4) #[B, C, H/32, W/32]

        return c5

def resnet18(pretrained=False, pretrained_pth=None):
    model = ResNet(BasicBlock, [2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], pretrained_pth), strict=False)
    return model

def resnet34(pretrained=False, pretrained_pth=None):
    model = ResNet(BasicBlock, [3,4,6,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], pretrained_pth), strict=False)
    return model 

def resnet50(pretrained=False, pretrained_pth=None):
    model = ResNet(Bottleneck, [3,4,6,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], pretrained_pth), strict=False)
    return model 

def resnet101(pretrained=False, pretrained_pth=None):
    model = ResNet(Bottleneck, [3,4,23,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], pretrained_pth), strict=False)
    return model 

def resnet152(pretrained=False, pretrained_pth=None):
    model = ResNet(Bottleneck, [3,8,36,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], pretrained_pth), strict=False)
    return model 
    
def build_backbone(model_name='resnet18', pretrained=False, pretrained_pth=None):
    if model_name == 'resnet18':
        model = resnet18(pretrained, pretrained_pth)
        feat_dim = 512
    elif model_name == 'resnet34':
        model = resnet34(pretrained, pretrained_pth)
        feat_dim = 512
    elif model_name == 'resnet50':
        model = resnet34(pretrained, pretrained_pth)
        feat_dim = 2048
    elif model_name == 'resnet101':
        model = resnet34(pretrained, pretrained_pth)
        feat_dim = 2048

    return model, feat_dim

if __name__ == "__main__":
    model, feat_dim = build_backbone(model_name='resnet18', pretrained=True)
    print(model)

    input = torch.randn(1, 3, 448, 448)
    output = model(input)
    print(output.size())
    print(output[0][0][0])
