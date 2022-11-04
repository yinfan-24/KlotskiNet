import torch.nn as nn
import torch.onnx


class Block(nn.Module):
    def __init__(self, in_channels, channels, stride, downsample=None):
        super(Block, self).__init__()
        # 1x1的卷积降维操作
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        # 3x3的卷积提取特征操作
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3),
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # 1x1的卷积升维操作
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=(1, 1),
                               bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # 4组卷积层的头一层网络会做一次降采样
        if self.downsample is not None:
            self.dconv = nn.Conv2d(in_channels, channels * 4, stride=stride, kernel_size=(1, 1), bias=False)
            self.dbn = nn.BatchNorm2d(channels * 4)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.dconv(identity)
            identity = self.dbn(identity)

        out += identity
        out = self.relu(out)

        return out

class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 对应第1组网络层，3*Resnet的基本结构
        self.conv64_1 = Block(64, 64, stride=1, downsample=True)
        self.conv64_2 = Block(256, 64, stride=1)
        self.conv64_3 = Block(256, 64, stride=1)
        # 对应第2组网络层，4*Resnet的基本结构
        self.conv128_1 = Block(256, 128, stride=2, downsample=True)
        self.conv128_2 = Block(128 * 4, 128, stride=1)
        self.conv128_3 = Block(128 * 4, 128, stride=1)
        self.conv128_4 = Block(128 * 4, 128, stride=1)
        # 对应第3组网络层，6*Resnet的基本结构
        self.conv256_1 = Block(512, 256, stride=2, downsample=True)
        self.conv256_2 = Block(256 * 4, 256, stride=1)
        self.conv256_3 = Block(256 * 4, 256, stride=1)
        self.conv256_4 = Block(256 * 4, 256, stride=1)
        self.conv256_5 = Block(256 * 4, 256, stride=1)
        self.conv256_6 = Block(256 * 4, 256, stride=1)
        # 对应第4组网络层，3*Resnet的基本结构
        self.conv512_1 = Block(1024, 512, stride=2, downsample=True)
        self.conv512_2 = Block(512 * 4, 512, stride=1)
        self.conv512_3 = Block(512 * 4, 512, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv64_1(x)
        x = self.conv64_2(x)
        x = self.conv64_3(x)
        x = self.conv128_1(x)
        x = self.conv128_2(x)
        x = self.conv128_3(x)
        x = self.conv128_4(x)
        x = self.conv256_1(x)
        x = self.conv256_2(x)
        x = self.conv256_3(x)
        x = self.conv256_4(x)
        x = self.conv256_5(x)
        x = self.conv256_6(x)
        x = self.conv512_1(x)
        x = self.conv512_2(x)
        x = self.conv512_3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def get_latent_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv64_1(x)
        x = self.conv64_2(x)
        x = self.conv64_3(x)
        x = self.conv128_1(x)
        x = self.conv128_2(x)
        x = self.conv128_3(x)
        x = self.conv128_4(x)
        x = self.conv256_1(x)
        x = self.conv256_2(x)
        x = self.conv256_3(x)
        x = self.conv256_4(x)
        x = self.conv256_5(x)
        x = self.conv256_6(x)
        x = self.conv512_1(x)
        x = self.conv512_2(x)
        x = self.conv512_3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x