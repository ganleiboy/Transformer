# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 使用resnet模型进行训练cifar10
# usage: python train-resnet18-cifar10.py
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet import ResNet18

# 超参数 ==========================================================================
data_dir = '../data/cifar10/'  # 需修改
best_model_path = "../output/resnet18-cifar10/best_model.pth"  # 需修改
latest_model_path = '../output/resnet18-cifar10/latest_model.pth'

num_epochs = 200  # 需修改
start_lr = 0.01
batch_size = 64*1  # 需修改，每个GPU上的图像数量×GPU数量

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = [i for i in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                    for x in ['train', "val"]}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                            shuffle=True, num_workers=12)
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
    best_acc = 0.0

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
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i%100 == 0:
                    print("{} epoch|ite:{}|{} Loss:{:.4f}".format(phase, epoch+1, i, loss/len(labels)))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('{} {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}'.format(timestamp, 
                    phase, epoch+1, num_epochs, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.module.state_dict())
                torch.save(best_model_wts, best_model_path)
        
        torch.save(model.module.state_dict(), latest_model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Train Acc: {:4f}'.format(best_acc))


# Finetuning, 构建网络
model = ResNet18()
model = model.cuda()
model = nn.DataParallel(model, device_ids=gpus)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

# Train
train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs)
print("Best model to ", best_model_path)