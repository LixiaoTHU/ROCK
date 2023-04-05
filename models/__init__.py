import mlconfig
import torch
from . import resnet
from . import resnext
from . import mresnext
from . import unet

mlconfig.register(resnet.ResNet50)
mlconfig.register(resnext.resnext101)
mlconfig.register(mresnext.mresnext101)
mlconfig.register(unet.UNet)