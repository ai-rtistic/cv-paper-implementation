import torch
import torch.nn as nn
from models.layers import Conv, Hourglass, Pool, Residual
from task.loss import HeatmapLoss

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Merge, self).__init__()
        self.conv = Conv(in_channels, out_channels, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, num_stack, in_channels, out_channels, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.num_stack = num_stack
        self.pre = nn.Sequential(  # pre layers
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, in_channels)
        )
        
        self.hgs = nn.ModuleList( [   # stacked hourglass
        nn.Sequential(
            Hourglass(4, in_channels, bn, increase),
        ) for i in range(num_stack)] )
        
        self.features = nn.ModuleList( [  
        nn.Sequential(
            Residual(in_channels, in_channels),
            Conv(in_channels, in_channels, 1, bn=True, relu=True)
        ) for i in range(num_stack)] )
        
        self.outs = nn.ModuleList( [Conv(in_channels, out_channels, 1, relu=False, bn=False) for i in range(num_stack)] )
        self.merge_features = nn.ModuleList( [Merge(in_channels, in_channels) for i in range(num_stack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(out_channels, in_channels) for i in range(num_stack-1)] )
        self.num_stack = num_stack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2) # imgs shape 변환 (Batch, H, W, C) -> (Batch, C, H, W)
        x = self.pre(x)
        combined_hm_preds = [] # 중간 레이어에서 나온 prediction 값을 모으기위한 리스트
        for i in range(self.num_stack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.num_stack - 1:  # if 마지막  hourglass 가 아니라면,
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature) # 중간레이어에서 x 에 conv 를 통과시킨 prediction 와 conv 를 통과시킨 feature 를 합쳐줌
        return torch.stack(combined_hm_preds, 1)  # dim=1 기준으로 쌓음

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.num_stack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss
