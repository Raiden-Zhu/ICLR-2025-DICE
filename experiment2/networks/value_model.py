import torch
import torch.nn as nn
import torch.nn.functional as F

class Value_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # 定义一个隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义输出层
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 通过隐藏层
        x = x.float()
        x = F.relu(self.fc1(x))
        # 通过输出层
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # 定义模型参数
    input_size = 1  # 输入是一个随机数
    hidden_size = 50  # 隐藏层的大小
    num_classes = 5  # 输出类别的数量

    # 创建模型实例
    model = Value_model(input_size, hidden_size, num_classes)

    # 打印模型结构
    print(model)