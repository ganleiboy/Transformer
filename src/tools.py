# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 一些小工具
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import matplotlib.pyplot as plt


def plot_logs(log_file, save_file):
    # 读取日志文件
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # 初始化列表用于存储数据
    epochs = []
    lr = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # 解析日志文件中的数据
    for line in lines:
        if line.startswith('train'):
            parts = line.split('|')
            epoch = int(parts[0].split(':')[1])
            lr_value = float(parts[1].split(':')[1])
            loss = float(parts[2].split(':')[1])
            acc = float(parts[3].split(':')[1])
            
            epochs.append(epoch)
            lr.append(lr_value)
            train_loss.append(loss)
            train_acc.append(acc)
        elif line.startswith('val'):
            parts = line.split('|')
            epoch = int(parts[0].split(':')[1])
            loss = float(parts[1].split(':')[1])
            acc = float(parts[2].split(':')[1])
            
            val_loss.append(loss)
            val_acc.append(acc)

    # 创建大图和5个子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 绘制lr图像
    axes[0, 0].plot(epochs, lr)
    axes[0, 0].set_title('Learning Rate')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('lr')

    # 绘制train loss图像
    axes[0, 1].plot(epochs, train_loss)
    axes[0, 1].set_title('Train Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')

    # 绘制train acc图像
    axes[0, 2].plot(epochs, train_acc)
    axes[0, 2].set_title('Train Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')

    # 绘制val loss图像
    axes[1, 0].plot(epochs, val_loss)
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')

    # 绘制val acc图像
    axes[1, 1].plot(epochs, val_acc)
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存大图
    plt.savefig(save_file)

    # 显示图像
    plt.show()


if __name__=="__main__":
    # 调用函数绘制图像
    logpath = "../output/resnet18-miniImagenet-02/train_log.txt"
    savepath = "../output/resnet18-miniImagenet-02/train_log.jpg"
    plot_logs(logpath, savepath)
    print("+"*30 + " end " + "+"*30)
