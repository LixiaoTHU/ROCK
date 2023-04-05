import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import time
import math

# Adapted from https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=32), # add group
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes, n_channels = 3, factor = 4, cls = "standard", normalize=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.normalize = normalize

        self.conv1 = nn.Conv2d(3, 64 * factor, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64 * factor)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down1 = Down(64 * factor, 128 * factor)
        self.down2 = Down(128 * factor, 256 * factor)
        self.down3 = Down(256 * factor, 512 * factor)
        self.up1 = Up(512 * factor, 256 * factor, bilinear=False)
        self.up2 = Up(256 * factor, 128 * factor, bilinear=False)
        self.cls = cls
        if self.cls == "standard":
            self.linear = nn.Linear(128 * factor, n_classes)
        elif self.cls == "segmentation":
            self.linear = nn.Conv2d(128 * factor, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        if self.normalize:
            x = (x - 0.5) / 0.5
        x1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        

        x = self.up1(x4, x3)
        x = self.up2(x, x2)

        out = x
        if self.cls == "standard":
            adaptiveAvgPoolWidth = out.shape[2]
            out = F.avg_pool2d(out, kernel_size=adaptiveAvgPoolWidth)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.cls == "segmentation":
            out = self.linear(out)
        return out