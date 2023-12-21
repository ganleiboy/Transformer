import os
import torch
import torch.nn as nn
import torchvision.models as models


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        # 基础的残差块
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)

        if stride == 2 or in_dim!=out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),  # 使用1x1和stride=1的卷积核进行下采样
                nn.BatchNorm2d(out_dim)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(h)
        x = identity + x
        x = self.relu(x)
        return x


class Resnet18(nn.Module):
    def __init__(self, in_dim=64, num_classes=10) -> None:
        super().__init__()
        # steam
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )
        # backbone
        self.layer1 = self._make_layer(num_block=2, in_dim=64, out_dim=64, stride=1)
        self.layer2 = self._make_layer(num_block=2, in_dim=64, out_dim=128, stride=2)
        self.layer3 = self._make_layer(num_block=2, in_dim=128, out_dim=256, stride=2)
        self.layer4 = self._make_layer(num_block=2, in_dim=256, out_dim=512, stride=2)

        # head
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512, out_features=num_classes)
        self.softmax = nn.Softmax(-1)  # 按行计算

    def _make_layer(self, num_block, in_dim, out_dim, stride):
        layer = []
        layer.append(ResidualBlock(in_dim=in_dim, out_dim=out_dim, stride=stride))  # 第一个block可能存在下采样
        for i in range(1, num_block):
            layer.append(ResidualBlock(in_dim=out_dim, out_dim=out_dim, stride=1))  # 后面的clock没有下采样
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # 交叉熵损失函数会自动对输入模型的预测值进行softmax计算。
        # refer: https://zhuanlan.zhihu.com/p/580367698?utm_id=0
        # x = self.softmax(x)  
        return x

    def residual(self):
        pass


class Resnet34(nn.Module):
    def __init__(self, in_dim=64, num_classes=10) -> None:
        super().__init__()
        # steam
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )
        # backbone
        self.layer1 = self._make_layer(num_block=3, in_dim=64, out_dim=64, stride=1)
        self.layer2 = self._make_layer(num_block=4, in_dim=64, out_dim=128, stride=2)
        self.layer3 = self._make_layer(num_block=6, in_dim=128, out_dim=256, stride=2)
        self.layer4 = self._make_layer(num_block=3, in_dim=256, out_dim=512, stride=2)

        # head
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512, out_features=num_classes)
        self.softmax = nn.Softmax(-1)  # 按行计算

    def _make_layer(self, num_block, in_dim, out_dim, stride):
        layer = []
        layer.append(ResidualBlock(in_dim=in_dim, out_dim=out_dim, stride=stride))  # 第一个block可能存在下采样
        for i in range(1, num_block):
            layer.append(ResidualBlock(in_dim=out_dim, out_dim=out_dim, stride=1))  # 后面的clock没有下采样
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # 交叉熵损失函数会自动对输入模型的预测值进行softmax计算。
        # refer: https://zhuanlan.zhihu.com/p/580367698?utm_id=0
        # x = self.softmax(x)  
        return x

    def residual(self):
        pass


def main():
    data = torch.rand([4,3,32,32])
    model = Resnet18()
    # model = models.resnet18()  # pytorch官方实现的版本
    # print(model)
    out = model(data)
    print(out.shape)
    # print(out)

if __name__ == "__main__":
    main()
    print("end.")
