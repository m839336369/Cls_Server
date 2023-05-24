import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from torchvision import datasets, transforms

labels = ['飞机', '手机', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
num_classes = 10 # 分类任务的类别数
num_epochs = 10 # 训练的轮
batch_size = 32 # 每个训练批次的样本数量
learning_rate = 0.001 # 学习率
def getInitModel(): # 用于获取初始化的ResNetCNN模型并进行训练。
    device = torch.device("cpu")
    model = ResNetCNN(num_classes=num_classes).to(device) # 创建一个ResNetCNN模型，并将其移动到所选设备（CPU）上
    if os.path.exists("my_model.pth"):
        load_model(model,"my_model.pth")
    train_loader = get_data_loaders(batch_size=batch_size) # 调用get_data_loaders函数获取训练数据加载器，用于加载训练数据并生成批次。

    criterion = nn.CrossEntropyLoss() # 定义交叉熵损失函数，用于计算模型的损失。
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 定义优化器，这里使用Adam优化器，并将模型的参数传递给优化器。

    for epoch in range(num_epochs): # 循环num_epochs次进行训练
        train(model, train_loader, optimizer, criterion, device) # 调用train函数，进行模型训练，传入模型、训练数据加载器、优化器、损失函数和设备
        save_model(model, path="my_model.pth") # 调用save_model函数保存模型的权重到文件"my_model.pth"
        print(f"Epoch {epoch + 1} completed.") # 打印当前轮次的完成信息
    return model



class ResNetCNN(nn.Module):# 定义了一个名为ResNetCNN的自定义模型类，继承自nn.Module，用于构建一个基于ResNet-50的卷积神经网络（CNN）模型。
    def __init__(self, num_classes=1000): # ResNetCNN类的构造函数__init__接受一个可选的num_classes参数，用于指定分类任务的类别数，默认为1000。
        super(ResNetCNN, self).__init__() # 确保正确初始化父类nn.Module。
        self.resnet = models.resnet50(pretrained=True) # 创建一个预训练的ResNet-50模型，其中pretrained=True表示加载预训练的权重。
        self.cnn = nn.Sequential( # 定义了一个包含一系列卷积、池化和批归一化层的顺序网络。这些层的结构和参数继承自预训练的ResNet-50模型。
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # 定义了一个二维卷积层，输入通道数为3（RGB图像），输出通道数为64，卷积核大小为7x7，步长为2，填充为3。
            nn.BatchNorm2d(64), # 定义了一个二维批归一化层，对卷积层的输出进行批归一化操作，以加速模型的训练和提高模型的鲁棒性。
            nn.ReLU(inplace=True), # 定义了一个ReLU激活函数层，对卷积层的输出进行非线性变换，引入非线性特征。
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # 定义了一个二维最大池化层，对卷积层的输出进行最大池化操作，以减小特征图的空间尺寸。
            self.resnet.layer1, # 四个主要残差
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)), # 定义了一个自适应平均池化层，对特征图进行自适应的平均池化操作，将特征图的尺寸调整为1x1。
        )
        self.fc = nn.Linear(2048, num_classes) # 定义了一个全连接（线性）层，将卷积部分的输出特征映射到指定的类别数。

    def forward(self, x): # 定义了模型的前向传播过程。它接受输入x，首先通过卷积部分self.cnn对输入进行特征提取，然后将特征展平为一维向量，最后通过全连接层self.fc生成最终的输出。
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)


def load_model(model, path="model.pth"):
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.eval()
    print("已加载预训练模型")


def train(model, train_loader, optimizer, criterion, device): # 用于在给定的训练数据上对模型进行一次训练迭代。它执行了模型的前向传播、损失计算、反向传播和参数更新等训练过程。
    model.train() # 将模型设置为训练模式，以启用训练相关的功能（如批量归一化和Dropout）。
    for batch_idx, (data, target) in enumerate(train_loader): # 迭代训练数据加载器，每次迭代返回一个批次的数据和目标标签。
        data, target = data.to(device), target.to(device) # 将数据和目标标签移动到指定的设备上，以便与模型在同一设备上进行计算。
        optimizer.zero_grad() # 将优化器的梯度缓冲区清零，以准备计算新一批数据的梯度。
        output = model(data) # 对输入数据进行前向传播，得到模型的输出
        loss = criterion(output, target) # 计算模型的损失，使用指定的损失函数计算模型的预测结果与实际标签之间的差异
        loss.backward() # 执行反向传播，计算梯度。
        optimizer.step() # 根据计算得到的梯度更新模型的参数，执行优化器的参数更新步骤。


def predict(model, data, device): # 对给定数据进行模型预测
    model.eval() # 将模型设置为评估模式，以禁用一些特定于训练的操作（如Dropout）。
    data = data.to(device) # 将输入数据移动到指定的设备上，以便与模型在同一设备上进行计算。
    output = model(data) # 前向传播，输出模型
    return torch.argmax(F.softmax(output, dim=1), dim=1).item() # F.softmax对模型的输出进行softmax归一化，以获得概率分布。
    # torch.argmax(..., dim=1).item()找到具有最高概率的类别索引，并将其转换为Python标量值。最后，函数返回预测的类别索引


def get_data_loaders(batch_size=64,train: bool = True): # 用于获取训练数据的数据加载器
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform) # 从CIFAR-10数据集中创建一个训练数据集对象，指定了数据存储路径、是否下载数据、图像变换等参数。

    # 设置读取的最大数量
    max_samples = 1024

    # 创建子集，选择指定数量的样本
    subset_indices = list(range(min(max_samples, len(dataset))))
    subset = Subset(dataset, subset_indices)

    train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True) # 创建一个训练数据的数据加载器，指定了训练数据集、批量大小和是否打乱数据。
    return train_loader



