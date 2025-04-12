# GoogleNet网络，
# 需要减少代码的冗余：
#   1. 使用函数
#   2. 把结构一样的结构块封装成一个类

import numpy as np
import matplotlib.pyplot as plt 
import torch.nn as nn # 引入nn模块，nn是torch的一个子模块，包含了很多神经网络的层、损失函数、优化器等
import torchvision
import torch
from torchvision import transforms # 针对图像进行原始处理
# Dataset是一个抽象类abstract class，不能实例化，只能被继承并自定义
# 类似一个冰箱，包含了所有数据（食材），但是不能直接使用
from torch.utils.data import Dataset
# DataLoader是一个数据加载器（shuffle、batch_size设置），可以实例化，并对数据进行批量处理
# 类似一个我的厨房助力，可以帮我取出食材（数据），准备好一批批分量供你使用，并且确保这个过程高效、顺畅
from torch.utils.data import DataLoader
import torch.nn.functional as F # 激活函数,引入relu
import torch.optim as optim # 优化器
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1. 构建数据集
batch_size = 64 # 批量大小,可以设置为128、256、512、1024等
# 一般是w * h * c,现在转换为c * w * h（主要是通道在前，提高计算效率）
# compose可以将()中的可调用的对象当作pipeline依次执行，
transforms = transforms.Compose(
    [   
        # Python处理图像的库现在一般是pillow，所以需要将原始图像的像素值(原来是0~255)转换为像素值（0,1）的图像张量tensor
        transforms.ToTensor(), # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,)) # 标准化，经验值：均值mean为0.1307，标准差std为0.3081
    ]
)
train_dataset = torchvision.datasets.MNIST(root='./data/MNIST', 
                                           train=True, 
                                           download=True, 
                                           transform=transforms)
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=3) 
test_dataset = torchvision.datasets.MNIST(root='./data/MNIST', 
                                           train=False, 
                                           download=True, 
                                           transform=transforms)
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False,
                         num_workers=3) 

# 2. 设计模型
class InceptionBlockA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlockA, self).__init__()
        # 分支1只有一个1x1卷积层,输出通道数为16(自定义)
        self.branch1_Conv1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 分支2有一个1x1卷积层和一个5x5卷积层,输出通道数分别为16和24(自定义)
        self.branch2_Conv1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2_Conv5x5 = nn.Conv2d(16, 24, kernel_size=5, padding=2)  # 填充为2，保持大小不变

        # 分支3有一个1x1卷积层和两个3x3卷积层,输出通道数分别为16、24和24(自定义)
        self.branch3_Conv1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3_Conv3x3_1 = nn.Conv2d(16, 24, kernel_size=3, padding=1)  # 填充为1，保持大小不变
        self.branch3_Conv3x3_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)  # 填充为1，保持大小不变

        # 分支4有一个Average Pooling层和一个1x1卷积层,输出通道数都为24(自定义)
        self.branch4_AvgPool = nn.Conv2d(in_channels, 24, kernel_size=1)  # 1x1卷积层

    def forward(self, x):
        # 分支1
        branch1 = self.branch1_Conv1x1(x)

        # 分支2
        branch2 = self.branch2_Conv1x1(x)
        branch2 = self.branch2_Conv5x5(branch2)

        # 分支3
        branch3 = self.branch3_Conv1x1(x)
        branch3 = self.branch3_Conv3x3_1(branch3)
        branch3 = self.branch3_Conv3x3_2(branch3)

        # 分支4
        branch4 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 使用平均池化层
        branch4 = self.branch4_AvgPool(branch4)  # 1x1卷积层

        # 拼接所有分支的输出
        outputs = [branch1, branch2, branch3, branch4]
        # 因为在(Batch, Channel, Height, Width)中，Channel在第二维，
        # 我们拼接是按照channel来拼接的，所以dim=1
        return torch.cat(outputs, dim=1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 输入通道数为1，输出通道数为10，卷积核大小为5x5

        # 定义Inception模块
        self.inception1 = InceptionBlockA(10)  # 输入通道数为10
        #inception1_out_channels = self._calculate_inception_output_channels(10)  # 动态计算inception1的输出通道数

        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # 输入通道数为inception1的输出通道数，输出通道数为20

        # 定义第二个Inception模块
        self.inception2 = InceptionBlockA(20)  # 输入通道数为conv2的输出通道数

        # 定义最大池化层
        self.mp = nn.MaxPool2d(2)  # 池化核大小为2x2

        # 动态计算展平后的数据长度
        self.flattened_size = self._get_flattened_size()

        # 定义全连接层
        self.fc = nn.Linear(1408, 10)

    # # 用于动态计算网络的输出通道数
    # def _calculate_inception_output_channels(self, in_channels):
    #     # 动态计算InceptionBlockA的输出通道数
    #     with torch.no_grad():
    #         x = torch.zeros(1, in_channels, 28, 28)  # 创建一个虚拟输入张量
    #         # 分支1
    #         branch1_out = self.inception1.branch1_Conv1x1(x).shape[1]  # 获取输出通道数(因为是bchw，c在1)
    #         # 分支2
    #         branch2_out = self.inception1.branch2_Conv5x5(
    #             self.inception1.branch2_Conv1x1(x)
    #         ).shape[1]
    #         # 分支3
    #         branch3_out = self.inception1.branch3_Conv3x3_2(
    #             self.inception1.branch3_Conv3x3_1(
    #                 self.inception1.branch3_Conv1x1(x)
    #             )
    #         ).shape[1]
    #         # 分支4
    #         branch4_out = self.inception1.branch4_Conv1x1(
    #             self.inception1.branch4_AvgPool(x)
    #         ).shape[1]
    #     channels_number = branch1_out + branch2_out + branch3_out + branch4_out
    #     return channels_number

    # 用于动态计算展平后的数据长度,是根据下面forward的过程来计算的
    def _get_flattened_size(self):
        # 创建一个虚拟输入张量，计算展平后的大小
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)  # MNIST图像大小为28x28，通道数为1
            x = F.relu(self.mp(self.conv1(x)))
            x = self.inception1(x)
            x = F.relu(self.mp(self.conv2(x)))
            x = self.inception2(x)
            return x.numel()  # 返回展平后的元素数量

    def forward(self, x):
        in_size = x.size(0)  # 获取batch_size,即样本数量n
                
        # 先卷基层,再池化层,最后relu激活
        x = F.relu(self.mp(self.conv1(x)))
        x = self.inception1(x)  # 使用Inception模块
        x = F.relu(self.mp(self.conv2(x)))
        x = self.inception2(x)
                
        # 展平数据以满足下面FCN需要的输入形式
        x = x.view(in_size, -1) 
                
        # 全连接层
        x = self.fc(x)  # 添加对最后一层的调用

        return x

model = Model()  # 确保在训练循环之前定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判断是否有GPU可用
model.to(device) # 将模型移动到GPU上进行训练,将权重、偏置等参数都转换为cuda类型的tensor


# 3. 构建损失函数loss和优化器optimizer对象(使用pytorch的API)
criterion = nn.CrossEntropyLoss()  
# 因为网络模型有点大，所以最好带动量避免局部最优
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 封装一轮训练函数
def train(epoch):
    training_losses = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
        # (1)获取数据，inputs是特征，targets是分类标签,都是tensor类型，都是张量
        # 由于循环中已经用元组表示，所以不需要再解包

        # 将数据移动到GPU上进行训练
        inputs, targets = inputs.to(device), targets.to(device)

        # 一定记得在前馈计算之前梯度清零，不然会累加
        optimizer.zero_grad()
        
        # (2)前馈过程算y^
        outputs = model(inputs)
        # (3)计算loss
        loss = criterion(outputs, targets)
        # (4)计算梯度
        loss.backward()
        # (5)更新权重
        optimizer.step()

        # 累加loss
        training_losses += loss.item() # 使用item()将loss转换为标量，不然会构建计算图

        # 每300个batch打印一次loss,不然输出太多计算成本增加
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, training_losses / 300))
            training_losses = 0.0

# 封装测试函数,只需算正向传播，不需要计算梯度（torch.no_grad）
def test():
    correct = 0
    total = 0
    with torch.no_grad():# 下面的代码不会再计算梯度
        for data in test_loader:
            # 获取数据
            images, labels = data
            # 将数据移动到GPU上进行训练
            images, labels = images.to(device), labels.to(device)
            # 前馈计算,预测计算y^
            outputs = model(images) 
            # 获取预测结果，求图片矩阵的每一行最大值的索引（下标）=>对应分类prediciton，_表示最大值
            _, predicted = torch.max(outputs.data, dim=1) # dim=1表示按行，dim=0表示按列
            # 累加总数
            total += labels.size(0) # labels.size(0)表示batch_size
            # 累加正确数
            correct += (predicted == labels).sum().item() # 使用item()将张量转换为标量
    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))

training_epochs = 10
# 需要进行封装，不然会报错
if __name__ == '__main__': 
    # 训练
    for epoch in range(training_epochs):
        train(epoch)
    # 测试
        test()