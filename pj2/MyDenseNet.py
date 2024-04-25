# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# DenseNet

class DenseBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        DROPOUT = 0.1
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(growth_rate)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = F.relu(self.dropout(self.bn(self.conv1(x))))
        # out = F.relu(self.dropout(self.bn(self.conv2(out))))
        out = torch.cat([x, out], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool(F.relu(self.bn(self.conv(x))))
        return out


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        in_channels = growth_rate  # Initial channels before first dense block

        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        # Dense Blocks
        self.dense1 = self._make_dense_block(DenseBlock, in_channels, block_config[0])
        in_channels += block_config[0] * growth_rate
        self.trans1 = self._make_transition(Transition, in_channels, in_channels // 2)
        in_channels = in_channels // 2

        self.dense2 = self._make_dense_block(DenseBlock, in_channels, block_config[1])
        in_channels += block_config[1] * growth_rate
        self.trans2 = self._make_transition(Transition, in_channels, in_channels // 2)
        in_channels = in_channels // 2

        self.dense3 = self._make_dense_block(DenseBlock, in_channels, block_config[2])
        in_channels += block_config[2] * growth_rate
        self.trans3 = self._make_transition(Transition, in_channels, in_channels // 2)
        in_channels = in_channels // 2

        self.dense4 = self._make_dense_block(DenseBlock, in_channels, block_config[3])
        in_channels += block_config[3] * growth_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(124, num_classes)

    def _make_dense_block(self, block, in_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def _make_transition(self, block, in_channels, out_channels):
        return block(in_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)

        out = F.relu(self.bn(out))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)


def design_model():
    return DenseNet(growth_rate=32, block_config=(2, 2, 2, 2), num_classes=10)
