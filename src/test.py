import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import resnet18  # 以 resnet18 为例
 

model = resnet18()
summary(model, (3, 224, 224), device="cpu")
print("end.")