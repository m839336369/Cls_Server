import torch
import numpy as np
from cnn_resnet import ResNetCNN, load_model, get_data_loaders


def evaluate_model(model, test_loader, num_classes): # 通过遍历test_loader获取测试数据集的每个批次。每个批次的图像数据和标签将移动到适当的设备上，然后通过模型进行预测。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # 获取每个样本预测结果的索引
            y_true.extend(labels.cpu().numpy()) # 真实标签和预测结果添加到y_true和y_pred列表中。
            y_pred.extend(predicted.cpu().numpy())
            for i in range(len(y_true)):
                print(f"true:{y_true[i]} predict:{y_pred[i]} result:{y_pred[i] == y_true[i]}") # 打印输出预测标签、真实标签、预测结果

    # 计算混淆矩阵
    cm = torch.zeros(num_classes, num_classes) # 大小为(num_classes, num_classes)的零张量
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # 计算准确度
    accuracy = torch.sum(torch.tensor(y_true) == torch.tensor(y_pred)).item() / len(y_true) # 转换为张量，在计算相等总数/样本总数
    print(accuracy)
    return accuracy, cm


test_loader = get_data_loaders(batch_size=128,train=False) # 调用get_data_loaders函数获取训练数据加载器，用于加载训练数据并生成批次。
device = torch.device("cpu")
model = ResNetCNN(10).to(device)  # 创建一个ResNetCNN模型，并将其移动到所选设备（CPU）上。
load_model(model, "./my_model_backpack.pth") # load_model函数用于加载模型权重，并将其应用于model对象.
evaluate_model(model,test_loader,10)

