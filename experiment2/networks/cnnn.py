import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

'''
class FlexibleCNN(nn.Module):
    def __init__(self):
        super(FlexibleCNN, self).__init__()
        self.conv1_3ch = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_1ch = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        if x.size(1) == 3:
            x = self.pool(F.relu(self.conv1_3ch(x)))
        elif x.size(1) == 1:
            x = self.pool(F.relu(self.conv1_1ch(x)))
        else:
            raise ValueError("Input channel must be 1 or 3")
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''
class FlexibleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FlexibleCNN, self).__init__()
        # 第一个卷积层：输入通道数为1（灰度图像），输出通道数为32，卷积核大小为3x3，填充为1
        self.conv1_3ch = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_1ch = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层：输入通道数为32，输出通道数为64，卷积核大小为3x3，填充为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 最大池化层：池化核大小为2x2，步幅为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一个全连接层：输入特征数为64*7*7（经过两次池化后，图像大小变为7x7），输出特征数为128
        self.fc1 = nn.Linear(64 * 7* 7, 128)
        # 第二个全连接层：输入特征数为128，输出特征数为10（MNIST数据集有10个类别）
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 第一层卷积 + 激活函数 + 池化
        if x.size(1) == 3:
            x = self.pool(F.relu(self.conv1_3ch(x)))
        elif x.size(1) == 1:
            x = self.pool(F.relu(self.conv1_1ch(x)))
        else:
            raise ValueError("Input channel must be 1 or 3")
        # 第二层卷积 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        # 展平特征图
        x = x.view(-1, 64 *7 * 7)
        # 第一层全连接 + 激活函数
        x = F.relu(self.fc1(x))
        # 第二层全连接
        x = self.fc2(x)
        return x