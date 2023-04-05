import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import time
import math

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, separate=False, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        padding = dilation
        if separate and stride==1 and dilation==1:
            self.conv2 = SeparableConv2d(planes, planes, kernel_size=3, stride=stride, 
                                padding=padding, bias=False, dilation = dilation)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                                padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion * planes)
        #     )
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cls = "standard", normalize=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.cls = cls
        self.normalize = normalize

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        if self.cls == "standard":
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        elif self.cls == "compare":
            self.layer3 = self._make_layer(block, 192, num_blocks[2], stride=1, dilation=2, separate = True)
            self.layer4 = self._make_layer(block, 384, num_blocks[3], stride=1, dilation=4, separate = True)
            self.linear = nn.Linear(384 * block.expansion, num_classes)
        elif self.cls == "segmentation":
            self.layer3 = self._make_layer(block, 192, num_blocks[2], stride=1, dilation=2, separate = True)
            self.layer4 = self._make_layer(block, 384, num_blocks[3], stride=1, dilation=4, separate = True)
            self.linear = nn.Conv2d(384 * block.expansion, num_classes, kernel_size=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1, separate=False):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, separate=separate))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize:
            x = (x - 0.5) / 0.5
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.cls == "standard" or self.cls == "compare":
            adaptiveAvgPoolWidth = out.shape[2]
            out = F.avg_pool2d(out, kernel_size=adaptiveAvgPoolWidth)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.cls == "segmentation":
            out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(cls, num_classes=10, normalize=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, cls=cls, normalize=normalize)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == "__main__":
    net = ResNet50(cls = "segmentation", num_classes=14)
    # print(net.get_logits_loss_grad_xent)
    # y = net(torch.randn(1, 3, 223, 224))
    inputs = torch.randn([1,3,224,224])
    flops, params = profile(net, (inputs,))
    print('flops: ', flops/1e9, 'params: ', params/1e6)
    # print(net)
    # print(.shape)
