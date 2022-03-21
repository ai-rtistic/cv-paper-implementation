import torch
import torch.nn as nn


Pool = nn.MaxPool2d

def batchnorm(x):    # Batch Normalization
    x = nn.BatchNorm2d(x.size()[1])(x)
    return x

class Conv(nn.Module):    # Convolutional layer
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        assert x.size()[1] == self.in_channels, f"Input channel mismatch, input channel:{x.size()[1]}, conv layer input channel:{self.in_channels}"
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):   # Residual block
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = Conv(in_channels, int(out_channels/2), 1, relu=False)  # bottlenecking
        self.bn2 = nn.BatchNorm2d(int(out_channels/2))
        self.conv2 = Conv(int(out_channels/2), int(out_channels/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_channels/2))
        self.conv3 = Conv(int(out_channels/2), out_channels, 1, relu=False)
        self.skip_layer = Conv(in_channels, out_channels, 1, relu=False)
        if in_channels == out_channels:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            identity = self.skip_layer(x)   # identity map channel 수 맞춰주는 과정
        else:
            identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += identity
        return out 


class Hourglass(nn.Module):
    def __init__(self, num_recursive, in_channels, bn=None, increase_channel=0):
        super(Hourglass, self).__init__()
        out_channels = in_channels + increase_channel
        self.up1 = Residual(in_channels, in_channels)
        # Lower branch
        self.pool1 = Pool(2, 2)   # height, width 이 반으로 줄어듦
        self.low1 = Residual(in_channels, out_channels)
        self.n = num_recursive
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(num_recursive-1, out_channels, bn=bn)
        else:
            self.low2 = Residual(out_channels, out_channels)
        self.low3 = Residual(out_channels, in_channels)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2