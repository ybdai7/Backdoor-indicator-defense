import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet
import numpy as np

class Generator(SimpleNet):

    def __init__(self, out_shape, in_shape):
        super(Generator, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
                *block(self.in_shape, 128, normalize=False),
                *block(128,256),
                *block(256,512),
                *block(512,1024),
                nn.Linear(1024,self.out_shape),
                nn.Tanh()
                )

    def forward(self, inp):
        out = self.model(inp)
        out = F.normalize(out, dim=1)
        return out
