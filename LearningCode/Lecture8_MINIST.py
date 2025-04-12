# 手写数据集MNIST的分类问题
import numpy as np
import matplotlib.pyplot as plt 
import torch
# 数据集和数据加载器
from torchvision import datasets, transforms
# DataLoader是一个数据加载器（shuffle、batch_size设置），可以实例化，并对数据进行批量处理
from torch.utils.data import DataLoader
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1. 构建数据集
# 训练集
train_dataset = datasets.MNIST(root='./data/MNIST', 
                               train=True, 
                               transform=transforms.ToTensor(), # 将PIL.Image或numpy.ndarray转换为tensor，并且归一化到[0,1]
                               download=True)# 如果数据集不存在，则下载数据集
# 测试集
test_dataset = datasets.MNIST(root='./data/MNIST', 
                              train=False, 
                              transform=transforms.ToTensor(), 
                              download=True)

# 分别构造两个数据加载器，用于训练和测试，不然内存会爆
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=32, 
                          shuffle=True)# 打乱数据，引入随机性，防止过拟合
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=32, 
                         shuffle=False)# 不需要打乱数据，每次测试数据顺序都是一样的，有利于评估模型

# 2. 设计模型（用于计算y^而不是计算loss）
# 激活函数还有很多：torch.nn.ReLU()不连续、torch.nn.Tanh()、torch.nn.LeakyReLU()、torch.nn.Softmax()、torch.nn.Softplus()、torch.nn.Softsign()、torch.nn.Softmin()、torch.nn.Softmax2d()、torch.nn.LogSoftmax()等
# 是激活函数和损失函数的选择，各种排列组合试试，哪个效果好就用哪个
class DiabetesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 输入是28*28的图片，输出是10个分类,需要先将图片展平成一维向量
        self.flatten = torch.nn.Flatten()
        
        # 构建三层神经网络
        # 第一层: 784 -> 512, ReLU激活
        self.linear1 = torch.nn.Linear(784, 512)
        self.relu1 = torch.nn.ReLU()
        
        # 第二层: 512 -> 128, ReLU激活 
        self.linear2 = torch.nn.Linear(512, 128)
        self.relu2 = torch.nn.ReLU()
        
        # 第三层: 128 -> 10, 输出10个分类
        self.linear3 = torch.nn.Linear(128, 10)
        # 最后一层不需要激活函数,因为后面会用CrossEntropyLoss,它内部包含了Softmax
        
        self.softmax = torch.nn.Softmax(dim=1)  # 添加Softmax激活函数用于多分类
        
    def forward(self, x):
        x = self.flatten(x) # 展平图片
        x = self.relu1(self.linear1(x)) # 第一层线性模型，然后进行ReLU激活
        x = self.relu2(self.linear2(x)) # 第二层线性模型，然后进行ReLU激活
        x = self.linear3(x) # 第三层线性模型，输出10个分类
        x = self.softmax(x) # 添加Softmax激活函数用于多分类
        return x

epochs = [] # 迭代次数
losses = [] # 损失值
training_epochs = 200

# 需要进行封装，不然会报错
if __name__ == '__main__': 
    model = DiabetesModel()  # 确保在训练循环之前定义模型
    # 3. 构建损失函数loss和优化器optimizer对象(使用pytorch的API)
    criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 实例化一个优化器对象，用于更新权重
    
    # 4. 训练training cycle（forward、backward、update）
    # 注意：如果结果没有收敛，可以增加迭代次数
    for epoch in range(training_epochs):
        epochs.append(epoch)
        total_loss = 0  # Initialize total loss for the epoch
        for batch_idx, (inputs, targets) in enumerate(train_loader, 0):#enumerate主要是为了获得当前迭代的次数
            # (1)获取数据，inputs是特征，targets是分类标签,都是tensor类型，都是张量
            # 由于循环中已经用元组表示，所以不需要再解包

            # (2)前馈过程算y^
            y_pred = model(inputs)
            print(targets.shape)  # Add this line to check the shape of targets
            loss = criterion(y_pred, targets)

            print(epoch, batch_idx, loss.item()) # 打印loss值

            # (3)反馈过程计算梯度
            optimizer.zero_grad() # 记住梯度要先清零!
            loss.backward()
            total_loss += loss.item()  # Accumulate loss for the epoch

            # (4)更新权重（链式法则）
            optimizer.step()

        # After all batches, calculate average loss for the epoch
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)  # Append average loss for the epoch

    # 可视化训练结果
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()  # 显示网格
    plt.show()