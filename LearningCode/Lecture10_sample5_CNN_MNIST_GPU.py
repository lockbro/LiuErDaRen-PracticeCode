# 使用CNN进行MINIST手写数字识别
# 输入图像的维度为（batch,1,28,28）
# 1、卷积层：卷积核设置为5x5，输入通道为1，输出通道为10，步长为1，填充为0
#    所以此时输出的维度为（batch,10,24,24），因为28-5+1=24，其中加一是因为步长为1
# 2、池化层：最大池化层，池化核设置为2x2，步长为2
#    所以此时输出的维度为（batch,10,12,12），因为24/2=12，其中除以2是因为步长为2，长和宽各减少一半
# 3、卷积层：卷积核设置为5x5，输入通道为10，输出通道为20，步长为1，填充为0
#    所以此时输出的维度为（batch,20,8,8），因为12-5+1=8，其中加一是因为步长为1
# 4、池化层：最大池化层，池化核设置为2x2，步长为2
#    所以此时输出的维度为（batch,20,4,4），因为8/2=4，其中除以2是因为步长为2，长和宽各减少一半
# 5、全连接层：将输出展平为1维，输入维度为20*4*4=320，输出维度为10(数字类别数0-9)
#    所以此时输出的维度为（batch,10），因为320/10=32，其中除以10是因为每个数字对应一个类别，最后输出10个数字的概率分布

# 一定要在整个过程中保证数据的维度是正确的，
# 卷基层和池华层不在乎输入数据的宽度和高度，只在乎输入数据的通道数和输出数据的通道数以及卷积核的大小和步长
# 最在乎数据长度和宽度的是：最后的分类器（全连接层），因为它需要将数据展平为1维
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
                          shuffle=True)
test_dataset = torchvision.datasets.MNIST(root='./data/MNIST', 
                                           train=False, 
                                           download=True, 
                                           transform=transforms)
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# 2. 设计模型（用于计算y^而不是计算loss）
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 因为没有权重，因此只需要一个pooling即可
        # 若有权重，则需要单独定义每一个池华层对象
        self.pooling = torch.nn.MaxPool2d(2)
        # 定义全连接层
        self.fc = torch.nn.Linear(320, 10)
        
    def forward(self, x):
        # 从(n,1,28,28)展开数据为(n,784) 
        batch_size = x.size(0) # 获取样本数量n作为batch_size
        
        # 先卷基层，再relu激活，最后池华层
        x = self.pooling(F.relu(self.conv1(x))) # 卷积层+池化层
        x = self.pooling(F.relu(self.conv2(x))) # 卷积层+池化层
        
        # 展平数据以满足下面FCN需要的输入形式，展平数据为(n,320)
        # -1表示自动计算维度
        x = x.view(batch_size, -1) 
        
        # 全连接层
        x = self.fc(x) 

        # 因为我们之后用的是CrossEntropyLoss损失函数，所以不需要激活函数
        return x

model = Model()  # 确保在训练循环之前定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判断是否有GPU可用
model.to(device) # 将模型移动到GPU上进行训练,将权重、偏置等参数都转换为cuda类型的tensor


# 3. 构建损失函数loss和优化器optimizer对象(使用pytorch的API)
criterion = torch.nn.CrossEntropyLoss()  
# 因为网络模型有点大，所以最好带动量避免局部最优
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

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