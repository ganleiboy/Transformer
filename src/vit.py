# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# refer：https://arxiv.org/abs/2010.11929
# usage: python vit.py
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.io import read_image
from PIL import Image

def test():
    # pytorch中相关的数据操作，教程中使用的是paddle_paddle
    data = torch.rand([3,2,4])  # 随机数在0~1之间
    data2 = torch.randint_like(data, 255).int()  # 通过*.int()修改数据类型
    data3 = torch.randint(0, 255, [3,2,4], dtype=torch.int)  # 通过dtype修改数据类型
    data4 = data3.transpose(1,0)  # 交换两个通道, 从(3,2,4)变为(2,3,4)
    data4_1 = data3.permute(1,0,2)  # 改变所有通道的顺序, data4_1和data4值相同
    data5 = data4.unsqueeze(-1)  # 从(2,3,4)变为(2,3,4,1)
    data6 = data5.squeeze(3)  # 从(2,3,4,1)变为(2,3,4)
    data7 = data6.reshape([4,6])  # reshape为(4,6)
    # 将data7沿最后一个维度切分成三块，切成三个(4,2)
    data8 = data7.chunk(3, -1)  # data8是tuple

    # refer:https://blog.csdn.net/qq_43665602/article/details/126281393/
    imgpath = "./data/724.jpg"
    img = Image.open(imgpath)
    img_arr = np.array(img)  # RGB, (h,w,c)
    img_tensor = torch.from_numpy(img_arr)  # numpy转tensor
    img_arr2 = img_tensor.numpy()  # tensor转numpy
    
    img2 = read_image(imgpath)  # torch.ByteTensor, torch.uint8, (c,h,w)
    print(img2.type())
    img3 = img2.float()  # torch.FloatTensor, torch.float32
    print(img3.type())


class Mlp(nn.Module):
    # mlp是Encoder中的Feed Forward Network模块
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim*mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim*mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channel, embed_dim, dropout=0.) -> None:
        super().__init__()
        # embed_dim可以理解为使用的卷积核的组数
        self.conv = nn.Conv2d(in_channel, embed_dim, patch_size, patch_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # x.shape=(batchsize, c, h, w)
        x = self.conv(x)  # x.shape=(batchsize, embed_dim, h', w')
        x = torch.flatten(x, 2)  # x.shape=(batchsize, embed_dim, h'*w')
        x = x.permute(0, 2, 1)   # x.shape=(batchsize, h'*w', embed_dim)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = Mlp(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        h = x
        x = self.norm(x)
        x = self.attn(x)
        x += h

        h = x
        x = self.norm(x)
        x = self.mlp(x)
        x += h
        return x


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)  # steam, embed_dim=16
        layer_list = [Encoder(16) for i in range(5)]
        self.encoders = nn.Sequential(*layer_list)  # encoder
        self.head = nn.Linear(16, 10)  # head, 10分类
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoders(x)  # [n, h'*w', c]
        # layernorm
        x = x.permute(0, 2, 1)  # [n, c, h'*w']
        x = self.avgpool(x)  # [n, c, 1]  # 简化计算，直接取均值
        x = torch.squeeze(x, 2)
        x = self.head(x)
        return x


def main():
    input = torch.randint(0, 255, [2,1,28,28]).float()  # (n,c,h,w)
    patch_embed = PatchEmbedding(image_size=28, patch_size=7, in_channel=1, embed_dim=1)
    out = patch_embed(input)
    # print(out)
    # print(out.shape)

    input = torch.randint(0, 255, [2,3,224,224]).float()  # (n,c,h,w)
    model = ViT()
    out = model(input)
    print(out)
    print(out.shape)
    print("+"*30 + " end " + "+"*30)

if __name__ == "__main__":
    main()
