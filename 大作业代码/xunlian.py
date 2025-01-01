from xc_code.model.SimpleCNN import SimpleCNN
from duqushuju import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def train_net(net, device, data_path, epochs=10, batch_size=8, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)  # C H W
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)  # B C h w

    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # 定义Loss算法
    criterion = nn.CrossEntropyLoss()

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 用来记录每个epoch的损失值
    epoch_losses = []

    # 训练epochs次
    for epoch in range(epochs):
        net.train()  # net.eval()  # 可以根据需要切换到评估模式
        print(f"Epoch [{epoch + 1}/{epochs}]")  # 打印当前轮次

        running_loss = 0.0  # 用来累积当前epoch的损失
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)  # CrossEntropyLoss需要标签为long类型

            pred = net(image)

            # 计算损失
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            running_loss += loss.item()  # 累加损失

            # 如果当前损失比历史最小损失小，则保存模型
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')

            # 反向传播
            loss.backward()
            optimizer.step()

        # 计算当前epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)  # 保存当前epoch的平均损失
        print(
            f"Epoch [{epoch + 1}/{epochs}] completed, Average Loss: {epoch_loss:.4f}, Best Loss: {best_loss.item():.4f}")

    # 训练完成后，画出损失函数下降曲线
    plt.plot(range(1, epochs + 1), epoch_losses, label="Training Loss", color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Function Decrease During Training')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络。
    net = SimpleCNN()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "CellData/chest_xray/train/"
    train_net(net, device, data_path)
    # print(torch.cuda.is_available())