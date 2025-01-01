import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np

class ISBI_Loader(Dataset):
    def __init__(self, data_path, target_size=(512, 512)):
        # 初始化函数，读取两个类别文件夹中的图片
        self.data_path = data_path  # 根目录路径，如data/train/
        self.target_size = target_size  # 目标尺寸 (height, width)

        # 通过glob读取两个文件夹的图片路径
        self.imgs_path = []
        self.labels = []
        self.class_labels = {'NORMAL': 0,
                             'PNEUMONIA': 1}

        # 遍历每个类别文件夹
        for class_name, class_id in self.class_labels.items():
            class_folder = os.path.join(data_path, class_name)
            img_paths = glob.glob(os.path.join(class_folder, '*.jpeg'))
            self.imgs_path.extend(img_paths)
            self.labels.extend([class_id] * len(img_paths))  # 为每个图像赋予对应的类别标签

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，flipCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]  # 图片路径
        label = self.labels[index]  # 对应的类别标签（整数形式）

        # 读取图片
        image = cv2.imread(image_path)

        # 将图像转换为灰度图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 调整图像大小为 target_size
        image = cv2.resize(image, self.target_size)  # 调整为目标尺寸 (target_size)
        # 将图像的形状调整为 (C, H, W)，即单通道 (1, H, W)
        image = image.reshape(1, image.shape[0], image.shape[1])  # C H W

        # 归一化图像像素值，将像素值从[0, 255]转为[0, 1]
        image = image / 255.0

        # 随机进行数据增强
        flipCode = random.choice([-1, 0, 1])
        image = self.augment(image, flipCode)

        # 返回图像和标签（标签是整数类型）
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        # 返回数据集大小
        return len(self.imgs_path)


# 展示一张图片和其对应的标签
def display_image_and_label(image, label):
    # 使用matplotlib展示图片和标签
    image = image.squeeze(0)  # 去除单通道维度 (C, H, W) -> (H, W)

    # 将标签转换为对应类别的名称
    label_names = ['NORMAL', 'PNEUMONIA']
    label_name = label_names[label]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('on')  # 显示坐标轴

    axes[1].imshow(image, cmap='gray')
    axes[1].set_title(f"Label: {label_name}")
    axes[1].axis('on')  # 显示坐标轴

    plt.show()


if __name__ == "__main__":
    # 数据路径
    isbi_dataset = ISBI_Loader("CellData/chest_xray/train/")

    print("数据个数：", len(isbi_dataset))

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1000,  # 设置批次大小
                                               shuffle=True)

    # 测试加载数据并展示
    for image, label in train_loader:
        print(f"Image batch shape: {image.shape}")
        print(f"Label batch shape: {label.shape}")

        # 打印标签
        print(f"Label (Integer): {label[0]}")  # 打印当前批次第一张图片的标签
        label_name = ['NORMAL', 'PNEUMONIA'][label[0]]
        print(f"Label Name: {label_name}")  # 打印对应的类别名称

        # 只展示当前批次的第一张图片
        display_image_and_label(image[0], label[0])

        break  # 只展示一个批次的第一张图片
