# U-Net

# 논문에서는 padding 을 사용하지 않아 input size 보다 output size 가 작아 overlap tile 기법을 사용함
# 본 구현에서는 padding 을 적용해 input size 와 output size 를 동일하게 적용 (성능 차이 별로 없음)


import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # contracting path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # expanding path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)  # feature map size 두 배로
                )
            self.ups.append(DoubleConv(feature*2, feature))

        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:  # 만약 x shape 이 skip connection (feature map) 의 size 와 안 맞을 시 (홀수 인 경우)
                x = TF.resize(x, size=skip_connection.shape[2:])  # skip connection 의 h, w 로 resize

            concat_skip_x = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip_x)
        
        x = self.final_conv(x)

        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 448, 448)
    model = UNet()
    out = model(x)

    print(x.shape)     # torch.Size([1, 3, 448, 448])
    print(out.shape)   # torch.Size([1, 1, 448, 448])












