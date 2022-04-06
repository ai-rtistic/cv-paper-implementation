import torch.nn as nn
import torch


class Yolov2(nn.Module):
    def __init__(self, num_classes,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)]): # grid cell 한칸을 기준으로 비율 (w, h) -> k-means 방법으로 결정한 최적의 amchor box 크기, 종횡비 (k=5)
        super(Yolov2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors

        # conv_block -> (in_channels, out_channels, filter size, stride, padding, maxpool = False(default))
        self.stage1_conv1 = self.conv_block(3, 32, 3, 1, 1, True)  # 416 x 416 x 3 -> 208 x 208 x 32   # input size 416 in paper
        self.stage1_conv2 = self.conv_block(32, 64, 3, 1, 1, True)   # 208 x 208 x 32 -> 104 x 104 x 64
        self.stage1_conv3 = self.conv_block(64, 128, 3, 1, 1)   # 104 x 104 x 64 -> 104 x 104 x 128
        self.stage1_conv4 = self.conv_block(128, 64, 1, 1, 0)   # 104 x 104 x 128 -> 104 x 104 x 64
        self.stage1_conv5 = self.conv_block(64, 128, 3, 1, 1, True)   # 104 x 104 x 64 -> 52 x52 x 128
        self.stage1_conv6 = self.conv_block(128, 256, 3, 1, 1)   # 52 x52 x 128 -> 52 x52 x 256
        self.stage1_conv7 = self.conv_block(256, 128, 1, 1, 0)   # 52 x52 x 256 -> 52 x52 x 128
        self.stage1_conv8 = self.conv_block(128, 256, 3, 1, 1, True)  # 52 x 52 x 128 -> 26 x 26 x 256
        self.stage1_conv9 = self.conv_block(256, 512, 3, 1, 1)   # 26 x 26 x 256 -> 26 x 26 x 512
        self.stage1_conv10 = self.conv_block(512, 256, 1, 1, 0)   # 26 x 26 x 512 -> 26 x 26 x 256                                
        self.stage1_conv11 = self.conv_block(256, 512, 3, 1, 1)   # 26 x 26 x 256 -> 26 x 26 x 512                               
        self.stage1_conv12 = self.conv_block(512, 256, 1, 1, 0)   # 26 x 26 x 512 -> 26 x 26 x 256
        self.stage1_conv13 = self.conv_block(256, 512, 3, 1, 1)   # 26 x 26 x 256 -> 26 x 26 x 512

        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)   #  26 x 26 x 512 ->  13 x 13 x 512

        self.stage2_a_conv1 = self.conv_block(512, 1024, 3, 1, 1)   # 13 x 13 x 512 -> 13 x 13 x 1024
        self.stage2_a_conv2 = self.conv_block(1024, 512, 1, 1, 0)   # 13 x 13 x 1024 -> 13 x 13 x 512
        self.stage2_a_conv3 = self.conv_block(512, 1024, 3, 1, 1)   # 13 x 13 x 512 -> 13 x 13 x 1024
        self.stage2_a_conv4 = self.conv_block(1024, 512, 1, 1, 0)   # 13 x 13 x 1024 -> 13 x 13 x 512
        self.stage2_a_conv5 = self.conv_block(512, 1024, 3, 1, 1)   # 13 x 13 x 512 -> 13 x 13 x 1024
        self.stage2_a_conv6 = self.conv_block(1024, 1024, 3, 1, 1)   # 13 x 13 x 1024 -> 13 x 13 x 1024
        self.stage2_a_conv7 = self.conv_block(1024, 1024, 1, 1, 0)   # 13 x 13 x 1024 -> 13 x 13 x 1024

        # self.stage2_b_conv = self.conv_block(512, 64, 1, 1, 0)
        
        self.stage3_conv1 = self.conv_block(2048+1024, 1024, 3, 1, 1)   # 13 x 13 x 3072 -> 13 x 13 x 1024
        self.stage3_conv2 = nn.Conv2d(1024, len(self.anchors) * (5 + num_classes), 1, 1, 0, bias=False)  # 13 x 13 x 1024 -> 13 x 13 x num anchor x (num class +5)

    def forward(self, x):
        out = self.stage1_conv1(x)
        out = self.stage1_conv2(out)
        out = self.stage1_conv3(out)
        out = self.stage1_conv4(out)
        out = self.stage1_conv5(out)
        out = self.stage1_conv6(out)
        out = self.stage1_conv7(out)
        out = self.stage1_conv8(out)
        out = self.stage1_conv9(out)
        out = self.stage1_conv10(out)
        out = self.stage1_conv11(out)
        out = self.stage1_conv12(out)
        out = self.stage1_conv13(out)

        residual = out   # passthrough layer

        out = self.stage2_a_maxpl(out)
        out = self.stage2_a_conv1(out)
        out = self.stage2_a_conv2(out)
        out = self.stage2_a_conv3(out)
        out = self.stage2_a_conv4(out)
        out = self.stage2_a_conv5(out)
        out = self.stage2_a_conv6(out)
        out = self.stage2_a_conv7(out)

        # out_2 = self.stage2_b_conv(residual)

        batch_size, num_channel, height, width = residual.data.size()    # 1, 512, 26, 26
        residual = residual.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        # (1, 512, 26, 26) -> (1, 128, 26, 2, 26, 2)
        residual = residual.permute(0, 3, 5, 1, 2, 4).contiguous()
        # (1, 128, 26, 2, 26, 2) -> (1, 2, 2, 128, 26, 26)
        residual = residual.view(batch_size, -1, int(height / 2), int(width / 2))
        # (1, 2, 2, 128, 26, 26) -> (1, 2048, 13, 13)
        out = torch.cat((out, residual), 1)   # out (1, 3072, 13, 13)
        out = self.stage3_conv1(out)  # out (1, 3072, 13, 13) -> (1, 1024, 13, 13)
        out = self.stage3_conv2(out)  # out (1, 1024, 13, 13) -> (1, 125, 13, 13)

        return out


    def conv_block(self, in_channels, out_channels, filter_size, stride, padding, maxpool=False):
        
        layers = [
            nn.Conv2d(in_channels, out_channels, filter_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        ]
        if maxpool:
            layers.append(
                nn.MaxPool2d(2,2)
            )
        conv_layers = nn.Sequential(*layers)
        
        return conv_layers



if __name__ == "__main__":
    # test
    net = Yolov2(20)
    x = torch.randn(1, 3, 416, 416)

    print(net(x).shape)   # 1, 125, 13, 13
