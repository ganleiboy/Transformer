# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 蒸馏学习
# refer: https://huaweicloud.csdn.net/63a56645b878a54545946373.html
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class StudentNet(nn.Module):
    # 一个很简单的网络
    def __init__(self, in_dim=64, num_classes=100):
        super().__init__()
        # steam
        self.conv1 = BasicConv(3, in_dim, stride=1)
        # backbone
        self.layer1 = BasicConv(in_dim, in_dim*2, stride=2)
        self.layer2 = BasicConv(in_dim*2, in_dim*4, stride=2)
        # head
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_dim*4, out_features=num_classes)
        # self.softmax = nn.Softmax(-1)  # 按行计算

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DistillationLoss:
    def __init__(self, alpha, temp):
        self.base_criterion = nn.CrossEntropyLoss()  # 基础的损失函数
        self.distill_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)  # 蒸馏学习的损失函数
        self.alpha = alpha
        self.temp = temp

    def __call__(self, student_out, labels, distill_type=None, teacher_out=None):
        # 先计算基础的交叉熵损失输出
        base_loss = self.base_criterion(student_out, labels)
        # 为none时不用考虑教师模型的输出
        if distill_type == None:
            return base_loss
        
        # 计算蒸馏损失
        if distill_type == "soft":
            # 直接计算二者的KL散度
            distill_loss = self.distill_criterion(F.log_softmax(student_out/self.temp, dim=1),
                                                  F.log_softmax(teacher_out/self.temp, dim=1))
        elif distill_type == "hard":
            # 将教师模型的输出进行one-hot处理，这样其实没有充分发挥蒸馏学习的作用
            distill_loss = self.base_criterion(student_out, teacher_out.argmax(dim=1))
        
        # 加权计算输出loss
        loss = (1 - self.alpha) * base_loss + self.alpha * distill_loss
        return loss


if __name__ == "__main__":
    stu = StudentNet()