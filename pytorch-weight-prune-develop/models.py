import torch
import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d 


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])
    

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = MaskedConv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(7*7*64, 10)
        
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        self.conv2.set_mask(torch.from_numpy(masks[1]))
        self.conv3.set_mask(torch.from_numpy(masks[2]))

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = MaskedConv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = MaskedConv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = MaskedConv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = MaskedConv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = MaskedConv2d(256, 256, kernel_size=3, padding=1)
        self.linear6 = MaskedLinear(256 * 6 * 6, 4096)
        self.linear7 = MaskedLinear(4096, 4096)
        self.linear8 = MaskedLinear(4096, num_classes)
        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.conv2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.conv3,
            nn.ReLU(inplace=True),
            self.conv4,
            nn.ReLU(inplace=True),
            self.conv5,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.linear6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.linear7,
            nn.ReLU(inplace=True),
            self.linear8,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        # self.conv1.set_mask(torch.from_numpy(masks[0]))
        # self.conv2.set_mask(torch.from_numpy(masks[1]))
        # self.conv4.set_mask(torch.from_numpy(masks[2]))
        # self.conv5.set_mask(torch.from_numpy(masks[3]))
        # self.linear6.set_mask(torch.from_numpy(masks[4]))
        # self.linear7.set_mask(torch.from_numpy(masks[5]))
        # self.linear8.set_mask(torch.from_numpy(masks[6]))
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.conv3.set_mask(masks[2])
        self.conv4.set_mask(masks[3])
        self.conv5.set_mask(masks[4])
        self.linear6.set_mask(masks[5])
        self.linear7.set_mask(masks[6])
        self.linear8.set_mask(masks[7])


class AdainNet(nn.Module):
    def __init__(self):
        super(AdainNet, self).__init__()

        
