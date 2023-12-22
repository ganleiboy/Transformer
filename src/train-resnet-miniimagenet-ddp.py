# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 使用DistributedDataParallel比DataParallel训练速度快一倍，10个epoch约5分钟
# refer：https://blog.csdn.net/weixin_43229348/article/details/124112404
# refer：https://blog.csdn.net/yaohaishen/article/details/127471992
# refer：https://blog.csdn.net/Komach/article/details/130765773
# usage：python -m torch.distributed.run --nproc_per_node=4 train-resnet18-miniimagenet-ddp.py
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3" # 尽量在torch被导入之前设置
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import distributed, DataLoader

from resnet import Resnet18
from tools import plot_logs


def save_log(str):
    global logpath
    with open(logpath, 'a', encoding="utf-8") as file:
        file.write(str)


def setup_distributed(rank, local_rank):
    # 设置分布式环境
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")  # 初始化进程组，使用NCCL作为通信后端
    device = torch.device("cuda", local_rank)  # 分布式训练只能使用GPU
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    return device


def define_network(device, local_rank, num_classes=100):
    # 定义网络架构
    net = Resnet18(num_classes=num_classes)
    net = net.to(device)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    return net


def define_dataloader(data_dir, batch_size, workers):
    # 创建数据加载器
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.2643, 0.2773, 0.3101], [0.2067, 0.2066, 0.2192])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.2643, 0.2773, 0.3101], [0.2067, 0.2066, 0.2192])
        ])
    }
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), data_transforms["val"])
    train_sampler = distributed.DistributedSampler(train_dataset, shuffle=True)  # 分布式训练使用分布式采样器
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=workers,
                            pin_memory=True,
                            sampler=train_sampler)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=workers, 
                            pin_memory=True)
    num_classes = len(train_dataset.classes)  # 自动计算类别数量
    return train_loader, val_loader, num_classes


def val(ep, net, val_loader, criterion, best_val_acc, best_epoch, best_model_path):
    net.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            val_total += targets.size(0)
            val_correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()
        
        epoch_acc = val_correct / val_total
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_epoch = ep
            best_model_wts = deepcopy(net.module.state_dict())
            torch.save(best_model_wts, best_model_path)
        
        save_log("val epoch:{} | loss:{:.6f} | acc:{:.4f}\n".format(ep, val_loss / len(val_loader), epoch_acc))
        print("   == val epoch: {} | loss: {:.3f} | acc: {:6.3f}%".format(
                ep, val_loss / len(val_loader), 100.0 * epoch_acc))
    return best_val_acc,best_epoch


def train(net, train_loader, val_loader, criterion, optimizer, rank, best_model_path, latest_model_path):
    since = time.time()
    best_model_wts   = deepcopy(net.module.state_dict())
    latest_model_wts = deepcopy(net.module.state_dict())
    best_val_acc = 0.0
    best_epoch = 1
    for ep in range(1, EPOCHS + 1):
        # 模型训练
        net.train()
        train_loss = correct = total = 0
        lr = 0.
        train_loader.sampler.set_epoch(ep)

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if rank == 0 and ((idx + 1) % 100 == 0 or (idx + 1) == len(train_loader)):
                print("   == train step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1, len(train_loader), ep, EPOCHS, train_loss / (idx + 1),
                        100.0 * correct / total))
        # 模型验证, 只在主卡上进行计算即可, 每张卡上计算结果几乎一样
        if rank == 0:
            save_log("train epoch:{} | lr:{:.6f} | loss:{:.6f} | acc:{:.4f}\n".format(ep, lr, train_loss/len(train_loader), correct / total))
            best_val_acc,best_epoch = val(ep, net, val_loader, criterion, best_val_acc, best_epoch, best_model_path)

    if rank == 0:
        torch.save(net.module.state_dict(), latest_model_path)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Val Acc: {:.4f} in epoch {}.'.format(best_val_acc, best_epoch))
        global logpath
        plot_logs(logpath, logpath.replace("txt", "jpg"))
        print("\n==============  Training Finished  ============== \n")


if __name__ == "__main__":
    BATCH_SIZE = 32 * 4  # bts=32*4时，每张卡显存占用8.5GB
    EPOCHS = 100
    WORKDERS = 4  # 实测设置为16或32时严重变慢，4比8要稍微快一点
    data_dir = '../data/miniImagenet/'
    save_folder = "../output/resnet18-miniImagenet-01/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    logpath = os.path.join(save_folder, "train_log.txt")
    best_model_path = os.path.join(save_folder, "best_model.pth")
    latest_model_path = os.path.join(save_folder, "latest_model.pth")
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = setup_distributed(rank, local_rank)
    train_loader, val_loader, num_classes = define_dataloader(data_dir, BATCH_SIZE, WORKDERS)
    net = define_network(device, local_rank, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)

    if rank == 0:
        print("==============  Start Training  ============== \n")

    train(net, train_loader, val_loader, criterion, optimizer, rank, best_model_path, latest_model_path)
    # print("+"*30 + " end " + "+"*30)