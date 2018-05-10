#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class Inception_a(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception_a, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n1x1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=n1x1),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n3x3red, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=n3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n3x3red, out_channels=n3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n3x3),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n5x5red, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n5x5red, out_channels=n5x5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n5x5, out_channels=n5x5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n5x5),
            nn.ReLU(inplace=True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_planes, out_channels=pool_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class Inception_b(nn.Module):
    def __init__(self, in_planes, n3x3red, n3x3, n5x5red, n5x5):
        super(Inception_b, self).__init__()

        # 1x1 conv -> 3x3 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n3x3red, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=n3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n3x3red, out_channels=n3x3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n3x3),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n5x5red, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n5x5red, out_channels=n5x5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n5x5, out_channels=n5x5, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n5x5),
            nn.ReLU(inplace=True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        return torch.cat([y1,y2,y3], 1)




class ldnet(nn.Module):
    def __init__(self):
        super(ldnet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=48),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=144, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=144),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            #nn.LocalResponseNorm(size, alpha=0.0001,beta=0.75,k=1)
            )

        self.3a = Inception_a(144, 48, 48, 48, 48, 72, 24)
        self.3b = Inception_a(192, 48, 48, 72, 48, 72, 48)
        self.3c = Inception_b(240, 96, 120, 48, 72)
        self.4a = Inception_a(432, 168, 48, 72, 72, 96, 96)
        self.4b = Inception_a(432, 144, 72, 96, 72, 96, 96)
        self.4c = Inception_a(432, 120, 96, 120, 96, 120, 96)
        self.4d = Inception_a(456, 72, 96, 144, 120, 144, 96)

        self.feature_up =nn.Sequential(
            nn.Conv2d(in_channels=456, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=73, kernel_size=1, stride=1, padding=0),


            )

        self.feature_down = nn.Sequential(
            nn.Conv2d(in_channels=456, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=72, kernel_size=1, stride=1, padding=0),

            )

        self.feature_cls = nn.Sequential(
            nn.Conv2d(in_channels=456, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, stride=1, padding=0),

            )

        self.feature_xushi(
            nn.Conv2d(in_channels=456, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, stride=1, padding=0),
            )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.3a(x)
        x = self.3b(x)
        x = self.3c(x)
        x = self.4a(x)
        x = self.4b(x)
        x = self.4c(x)
        x = self.4d(x)

        feature_up = self.feature_up(x)
        feature_down = self.feature_down(x)
        feature_cls = self.feature_cls(x)
        feature_xushi = self.feature_xushi(x)
        feature_cls = self.softmax(feature_cls)
        feature_xushi = self.softmax(feature_xushi)

        return [feature_up, feature_down, feature_cls, feature_xushi]


