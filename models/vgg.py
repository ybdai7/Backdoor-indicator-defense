import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet

class VGG(SimpleNet):
    def __init__(self, vgg, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg)
        self.dense = nn.Sequential(
                nn.Linear(512,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                )
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    def _make_layers(self,vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                          nn.BatchNorm2d(x),
                          nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class SupConVGG(SimpleNet):
    def __init__(self, vgg):
        super(SupConVGG, self).__init__()
        self.features = self._make_layers(vgg)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out,1)
        out = F.normalize(out, dim=1)
        return out

    def _make_layers(self,vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                          nn.BatchNorm2d(x),
                          nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG16(num_classes):
    vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(vgg16, num_classes=num_classes)

def SupConVGG16():
    vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return SupConVGG(vgg16)

def VGG19(num_classes):
    vgg19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,  'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return VGG(vgg19, num_class=num_classes)

def SupConVGG19():
    vgg19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,  'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return SupConVGG(vgg19)
