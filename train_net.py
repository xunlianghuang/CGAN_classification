# coding=utf-8
import torch
from torch import nn
from torch.utils.data import DataLoader
from util import train
from dataset import  SIGNSDataset
from net import Network
import torchvision.transforms as transforms
train_transformer = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor
eval_transformer = transforms.Compose([
    transforms.ToTensor()])
dataset_train = SIGNSDataset(data_dir="data/train",transform=train_transformer,augmentation=True)
dataset_test = SIGNSDataset(data_dir="data/test",transform=eval_transformer,augmentation=False)

# 使用预训练模型
LR = 0.01
EPOCHES = 50
BATCH_SIZE = 256
# 使用DataLoader定义迭代器
train_data = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_data = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
net = Network(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
print("开始训练")
# 开始训练
train(net, train_data, valid_data, 10, optimizer, criterion)