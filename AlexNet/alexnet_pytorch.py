import torch
import torch.nn as nn
import torchvision

# AlexNet Architecture
# 3 x 224 x 224 input -> s

class AlexNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), # (N, 96, 55, 55)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # (N, 96, 27, 27)
            # nn.LocalResponseNorm()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # (N, 256, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # (N, 256, 13, 13)
            # nn.LocalResponseNorm()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # (N, 384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), # (N, 384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # (N, 384, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (N, 256, 6, 6)
            nn.Flatten() # (N, 256*6*6)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096), # (N, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), # (N, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes) # (N, n_classes)
        )

        # Weight Initialization -> Xavier initialization
        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)

        return out


    def init_weights(self, layer):
            if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)


if __name__ == '__main__':
    x = torch.randn(128,3,224,224)
    
    model_custom = AlexNet(n_classes=1000)
    print(model_custom(x).shape)  #  output : 128 x 1000

    model_torch = torchvision.models.alexnet()
    print(model_torch(x).shape)  #  output : 128 x 1000
