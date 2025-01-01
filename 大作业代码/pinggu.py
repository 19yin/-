from xc_code.model.SimpleCNN import SimpleCNN
from duqushuju import ISBI_Loader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns



def cs(net, device, data_path, batch_size=10, delay=0.5):
    # 加载测试集
    isbi_dataset = ISBI_Loader(data_path)  # C H W
    test_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)  # B C H W

    # 将网络拷贝到device中
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))

    # 测试模式
    net.eval()  # net.train() 训练模式，net.eval() 测试模式

    # 存储所有的预测标签与真实标签
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in test_loader:
            # 加入延迟来控制频率
            time.sleep(delay)  # 每个批次处理后加入延迟
            # 将图像和标签转移到设备上
            images = images.float().to(device)  # 转换为 float32
            labels = labels.to(device)

            # 获取模型输出（预测值）
            outputs = net(images)

            # 取最大概率的类别作为预测值
            _, predicted = torch.max(outputs, 1)

            # 将预测结果与真实标签收集
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 可视化前几个预测图像
            for i in range(min(1, len(images))):  # 可视化图像
                img = images[i].cpu().numpy().squeeze()  # 从tensor中提取出图像，并移除多余的维度
                label = labels[i].item()  # 获取真实标签
                pred = predicted[i].item()  # 获取预测标签

                # 绘制图像
                plt.imshow(img, cmap='gray')
                plt.title(f"True: {label}, Pred: {pred}")
                plt.show()
                break

    # 输出整体准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 计算其他评估指标
    precision = precision_score(all_labels, all_preds, average='weighted')  # 加权精确度
    recall = recall_score(all_labels, all_preds, average='weighted')  # 加权召回率
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 加权F1分数

    print(f"Test Precision: {precision * 100:.2f}%")
    print(f"Test Recall: {recall * 100:.2f}%")
    print(f"Test F1-Score: {f1 * 100:.2f}%")

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")

    # 可视化混淆矩阵
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_labels),
                yticklabels=np.unique(all_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络
    net = SimpleCNN()

    # 将网络拷贝到device中
    net.to(device=device)

    # 指定测试集地址，开始测试
    data_path = "CellData/chest_xray/test/"  # 根据你的数据集路径修改
    cs(net, device, data_path)
