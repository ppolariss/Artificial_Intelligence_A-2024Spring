# Importing Libraries
import torch
import torch.nn as nn


# resnext

class Block(nn.Module):
    def __init__(self, in_channels, cardinality=32, bottleneck_width=4):
        super(Block, self).__init__()
        width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(width),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(num_blocks[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        # 512 * cardinality * bottleneck_width

    def _make_layer(self, num_blocks, planes, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                Block(self.in_channels, self.cardinality, self.bottleneck_width)
            )
            self.in_channels = 128
            # planes * self.cardinality * self.bottleneck_width
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnext50_32x4d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4)