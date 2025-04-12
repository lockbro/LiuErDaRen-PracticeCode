# 使用CNN进行MINIST手写数字识别
# 输入图像的维度为（batch,1,28,28）
# 作业：尝试3个卷基层、3个ReLu层，3个池华层，3个全连接层

import numpy as np
import matplotlib.pyplot as plt 
import torch
import torchvision
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

# 2. 设计模型（用于计算y^而不是计算loss）
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义卷积层
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)  # 改为 3x3 卷积核
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)  # 改为 3x3 卷积核 
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3)  # 改为 3x3 卷积核
        # 定义池化层
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，2x2窗口
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化，3x3窗口，步幅为2，带填充
        self.pooling3 = torch.nn.MaxPool2d(kernel_size=2, stride=1)  # 最大池化，2x2窗口，步幅为1
        # 动态计算展平后的数据长度
        self.flattened_size = self._get_flattened_size()

        # 定义全连接层
        self.fc1 = torch.nn.Linear(self.flattened_size, 160)
        self.fc2 = torch.nn.Linear(160, 80)
        self.fc3 = torch.nn.Linear(80, 10)

    def _get_flattened_size(self):
        # 创建一个虚拟输入张量，计算展平后的大小
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)  # 假设输入是 (1, 1, 28, 28)
            x = self.pooling1(F.relu(self.conv1(x)))
            x = self.pooling2(F.relu(self.conv2(x)))
            x = self.pooling3(F.relu(self.conv3(x)))
            return x.numel()  # 返回展平后的元素数量

    def forward(self, x):
        batch_size = x.size(0)  # 获取样本数量n作为batch_size
        
        # 先卷基层，再relu激活，最后池化层
        x = self.pooling1(F.relu(self.conv1(x)))
        x = self.pooling2(F.relu(self.conv2(x))) 
        x = self.pooling3(F.relu(self.conv3(x)))
        
        # 展平数据以满足下面FCN需要的输入形式
        x = x.view(batch_size, -1) 
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 添加对最后一层的调用

        return x

model = Model()  # 确保在训练循环之前定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判断是否有GPU可用
model.to(device) # 将模型移动到GPU上进行训练,将权重、偏置等参数都转换为cuda类型的tensor


# 3. 构建损失函数loss和优化器optimizer对象(使用pytorch的API)
criterion = torch.nn.CrossEntropyLoss()  
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