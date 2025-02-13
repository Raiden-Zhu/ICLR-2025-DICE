import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_channels=1, image_size=28, num_classes=10):
        super(MLP, self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        
        # 计算输入特征的数量
        input_features = input_channels * image_size * image_size
        
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 将输入展平为二维张量
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Simple(nn.Module):
    def __init__(self, input_channels=1, image_size=28, num_classes=10):
        super(Simple, self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        
        # 计算输入特征的数量
        input_features = input_channels * image_size * image_size
        
        self.fc1 = nn.Linear(input_features, num_classes)
        

    def forward(self, x):
        # 将输入展平为二维张量
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    
class ImagenetMLP(nn.Module):
    def __init__(self, input_channels=1, image_size=28, num_classes=10):
        super(ImagenetMLP, self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        
        # 计算输入特征的数量
        input_features = input_channels * image_size * image_size
        
        self.fc1 = nn.Linear(input_features, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 将输入展平为二维张量
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
