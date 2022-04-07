import torch
import torch.nn as nn


class residualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += self.shortcut(identity)  # identity mapping

        x = self.relu(x)
        return x


class ResnetFPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResnetFPN, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, num_blocks[0], 64, stride=1)
        self.layer2 = self._make_layer(block, num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(block, num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(block, num_blocks[3], 512, stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Bottom-up pathway
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = x   # 64 channels  # don't use c1 in paper
        
        x = self.maxpool(x)
        x = self.layer1(x)
        c2 = x   # 256 channels

        x = self.layer2(x)
        c3 = x   # 512 channels

        x = self.layer3(x)
        c4 = x   # 1024 channels

        x = self.layer4(x)
        c5 = x   # 2048 channels

        # Top-down pathwaty
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5

    def _make_layer(self, residual_block, num_blocks, out_channels, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(residual_block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * residual_block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        upsample = nn.Upsample(size=(H,W), mode='nearest')
        return upsample(x) + y


def ResNet50_FPN():
    return ResnetFPN(residualBlock, [3, 4, 6, 3])

def ResNet101_FPN():
    return ResnetFPN(residualBlock, [3, 4, 23, 3])

def ResNet152_FPN():
    return ResnetFPN(residualBlock, [3, 8, 36, 3])



if __name__=='__main__':

    net = ResNet50_FPN()
    x = torch.randn(1, 3, 448, 448)
    ps = net(x)
    for p in ps:
        print(p.size())
        # torch.Size([1, 256, 112, 112])
        # torch.Size([1, 256, 56, 56])
        # torch.Size([1, 256, 28, 28])
        # torch.Size([1, 256, 14, 14])