# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# refer：https://arxiv.org/abs/2010.11929
# usage: python vit.py
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.io import read_image
from PIL import Image


class PatchEmbedding(nn.Module):
    # 整个过程的输出和Transformer中的词嵌入是相同的，都是(batchsize, words_num, embed_dim)
    def __init__(self, image_size, patch_size, in_channel, embed_dim, dropout=0.) -> None:
        """
        # embed_dim是词嵌入维度，可以理解为使用的卷积核的组数
        # in_channel是输入图像的通道数
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channel, embed_dim, patch_size, patch_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # x.shape=(batchsize, c, h, w)
        x = self.conv(x)  # x.shape=(batchsize, embed_dim, h', w')
        x = torch.flatten(x, 2)  # x.shape=(batchsize, embed_dim, h'*w')
        x = x.permute(0, 2, 1)   # x.shape=(batchsize, h'*w', embed_dim)
        x = self.dropout(x)
        return x


class FFN(nn.Module):
    # FFN是Encoder中的Feed Forward Network模块，输入输出尺寸不变
    def __init__(self, embed_dim, FFN_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim*FFN_ratio))
        self.fc2 = nn.Linear(int(embed_dim*FFN_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.fc1(x)  # 先升维提取特征
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Identity()  # 先使用恒等映射替换注意力模块，简化问题
        self.FFN = FFN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        h = x
        x = self.norm(x)  # 很多论文都说是norm前置效果更好一些
        x = self.attn(x)
        x += h

        h = x
        x = self.norm(x)
        x = self.FFN(x)
        x += h
        return x


class Head(nn.Module):
    def __init__(self, embed_dim, class_num):
        super().__init__()
        self.linear = nn.Linear(embed_dim, class_num)  # head, 映射为10分类，cifar10
        self.softmax = nn.Softmax(dim=1)  # dim=1，按行计算
    
    def forward(self, x):
        return self.softmax(self.linear(x))


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)  # steam, embed_dim=16
        layer_list = [Encoder(16) for i in range(5)]
        self.encoders = nn.Sequential(*layer_list)  # encoder
        self.head = Head(embed_dim=16, class_num=10)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoders(x)  # [n, h'*w', c]，此处的c就是embed_dim，类似于CNN中的channel
        x = x.permute(0, 2, 1)  # [n, c, h'*w']
        x = self.avgpool(x)  # [n, c, 1]  # 简化计算，直接取均值。类似于CNN中每个channel计算全局平均池化
        x = torch.squeeze(x, 2)  # [n, c]
        x = self.head(x)  # [n, 10]
        return x


def main():
    input = torch.randint(0, 255, [2,1,28,28]).float()  # (n,c,h,w)，灰度图
    patch_embed = PatchEmbedding(image_size=28, patch_size=7, in_channel=1, embed_dim=16)
    out = patch_embed(input)
    print(out)
    print(out.shape)
    print("+"*50)

    input = torch.randint(0, 255, [2,3,224,224]).float()  # (n,c,h,w)，彩图
    model = ViT()
    out = model(input)  # [2, 10]
    print(out)
    print(out.shape)
    print("end.")

if __name__ == "__main__":
    main()
