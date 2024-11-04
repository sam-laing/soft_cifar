import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
import os
import sys
import time
import random

from resnet import make_resnet_cifar
import torch.optim as optim
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


num_epochs = 200
learning_rate=0.07

train_mean = [0.4914, 0.4822, 0.4465]
train_std = [0.2023, 0.1994, 0.2010]
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root="/mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py",
                                             train=True,
                                             transform=train_transform,
                                            )

test_dataset = torchvision.datasets.CIFAR10(root='/mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py',
                                            train=False,
                                            transform=test_transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=False)




# Training the model
def train(num_epochs, train_loader, model, criterion, optimizer, scheduler, device, l2_reg=0):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss
            correct += (torch.argmax(outputs, 1) == labels).sum().item()
            total += images.shape[0]

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Accuracy: ", correct / total, " Train Loss ", train_loss / total )
        scheduler.step()

# Testing the model
def test(test_loader, model, device):
    model.eval()  # Eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            correct += (torch.argmax(outputs, 1) == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total}%')


if __name__ == "__main__":

    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4 )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65, 120, 165], gamma=0.5)


    train(num_epochs, train_loader, model, criterion, optimizer, scheduler, device)
    test(test_loader, model, device)
    

    path = "/mnt/qb/work/oh/owl886/soft_cifar/models/"
    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
    torch.save(model.state_dict(), path + f"{SLURM_JOB_ID}_cifar10full1.pth")
