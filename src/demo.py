# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 加载预训练模型
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from resnet import ResNet18

# 超参数 ====================================================================
best_model_path = "./best_model.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_backbone(model, pretrained_model, device):
    """
    逐层加载预训练模型的参数到model中。
    https://blog.csdn.net/Charles5101/article/details/101028435
    """
    backbone = torch.load(pretrained_model, map_location=device)  # torch.device('cpu')
    layer = 0
    new_dict = {}
    model_dict = model.state_dict()
    
    for key, value in backbone.items():        
        new_dict[key] = value
        print("[{}: load key from backbone to model: {}]".format(layer, key))
        layer += 1
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)


# 加载预训练模型
model_pre = torch.load(best_model_path, map_location=device)
print(model_pre.keys())  # 查看有哪些key

# 构建新模型
model = ResNet18()
model = model.cuda()
print(model.state_dict().keys())  # 查看有哪些key

# 加载预训练模型的参数到新模型
# model.load_state_dict(model_pre)  # 由于网络结构变了，所以加载失败，会报错
load_backbone(model, model_pre, device)  # 只加载可以对应上的层，加载成功

print("+"*30)