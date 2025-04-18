import torch
import torch.nn as nn
import torch.nn.functional as F

class Value_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
     
        self.fc1 = nn.Linear(input_size, hidden_size)
  
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
    
        x = x.float()
        x = F.relu(self.fc1(x))
      
        x = self.fc2(x)
        return x

if __name__ == '__main__':
 
    input_size = 1  
    hidden_size = 50 
    num_classes = 5

    # 创建模型实例
    model = Value_model(input_size, hidden_size, num_classes)

    # 打印模型结构
    print(model)