import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1通道输入（灰度图）
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.fc1 = nn.Linear(128 * 64 * 64, 512)  # 根据输入图像的尺寸调整
        self.fc2 = nn.Linear(512, 2)  # 输出2个类别：NORMAL和PNEUMONIA

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 64 * 64)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
