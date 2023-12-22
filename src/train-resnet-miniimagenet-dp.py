# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 使用resnet模型进行训练miniImageNet
# 使用DataParallel训练速度要慢很多（一半），10个epoch约8分钟；但单卡占用的显存少，可以使用更大的batchsize（2倍）
# https://zhuanlan.zhihu.com/p/113694038?utm_id=0&wd=&eqid=98b9f81a000c0f36000000046466449a
# usage: python train-resnet18-miniimagenet-dp.py
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet import Resnet18
import multiprocessing
# from apex import amp

torch.backends.cudnn.benchmark = True  # 设置前后好像没啥区别

# 参数 ==========================================================================
data_dir = '../data/miniImagenet/'  # 需修改
save_folder = "../output/resnet18-miniImagenet-01/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
best_model_path = os.path.join(save_folder, "best_model.pth")
latest_model_path = os.path.join(save_folder, "latest_model.pth")

num_epochs = 60  # 需修改
batch_size = 64*4  # 需修改，单卡图像数量×GPU数量
start_lr = 0.01
num_workers = 4  # 实测设置为16或32时严重变慢，4比8要稍微快一点

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
gpus = [i for i in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集的transform
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

# ImageFolder: 按文件夹给数据分类，一个文件夹为一类，label会自动标注好
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', "val"]}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
               for x in ['train', "val"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', "val"]}

class_names = image_datasets['train'].classes  # list
num_classes = len(class_names)

# define training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    print("start training...")
    since = time.time()

    best_model_wts = copy.deepcopy(model.module.state_dict())
    latest_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_acc = 0.0
    best_epoch = 1

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', "val"]:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i%100 == 0:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("{} {} epoch|ite:{}|{} Loss:{:.4f}".format(timestamp, phase, epoch+1, i, loss/len(labels)))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('{} {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}'.format(timestamp, phase, epoch+1, num_epochs, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.module.state_dict())
                torch.save(best_model_wts, best_model_path)
        
        torch.save(model.module.state_dict(), latest_model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {:4f} in epoch {}'.format(best_val_acc, best_epoch))


# Finetuning, 构建网络
model = Resnet18(num_classes=num_classes)
model = model.to(device)
model = nn.DataParallel(model, device_ids=gpus)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Train
train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs)
print("Best model to ", best_model_path)
print("+"*30 + " end " + "+"*30)