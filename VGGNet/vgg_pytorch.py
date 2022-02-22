import torch
import torch.nn as nn
import torchvision



arc = {
	'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}





class VGGNet(nn.Module):
	def __init__(self, in_channels=3, num_classes=1000, architecture='VGG16'):
		super(VGGNet, self).__init__()
		self.in_channels = in_channels
		self.conv_layers = self.create_conv_layers(arc[architecture])
		self.fc_layers = nn.Sequential(
			nn.Linear(512*7*7, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(4096, num_classes)
		)
		
		
	def forward(self, x):
		x = self.conv_layers(x)
		x = x.reshape(x.shape[0], -1)  # (N, 512, 7, 7) -> (N, 512*7*7)
		x = self.fc_layers(x)
		return x



	def create_conv_layers(self, architecture):
		layers = []
		in_channels = self.in_channels

		for x in architecture:
			if type(x) == int:
				out_channels = x
				layers += [
					nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
					nn.BatchNorm2d(x), # Batch Normalization 은 논문에는 없지만 성능 향상을 위해 추가
					nn.ReLU(inplace=True)
					]
				in_channels = x
			elif x == "M":
				layers += [nn.MaxPool2d(kernel_size=(2,2), stride=2)]
		
		return nn.Sequential(*layers)


if __name__ == '__main__':			
    x= torch.randn(1, 3, 224, 224)

    model_custom_16 = VGGNet(architecture='VGG16')
    print(model_custom_16(x).shape)

    model_torch_16 = torchvision.models.vgg16_bn()
    print(model_torch_16(x).shape)

    model_custom_19 = VGGNet(architecture='VGG19')
    print(model_custom_19(x).shape)

    model_torch_19 = torchvision.models.vgg19_bn()
    print(model_torch_19(x).shape)







