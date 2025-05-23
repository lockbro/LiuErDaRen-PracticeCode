# 使用pytorch处理糖尿病数据
import numpy as np
import matplotlib.pyplot as plt # 该模块主要用于绘制 2D 图表
import torch
import torch.nn.functional as F # 该模块包含了很多激活函数和损失函数
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1. 构建数据集
# 读取数据集，delimiter表示分隔符，dtype表示数据类型为float32
dataset = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32) 
# torch.from_numpy(ndarray)函数将numpy数据转换为tensor数据
# 通用形式： start:end:step
x_data = torch.from_numpy(dataset[:,:-1]) # 读取特征数据,切片操作，":表示所有行,:-1表示除最后一列之外的所有列",读取除最后一列之外的所有列
y_data = torch.from_numpy(dataset[:, [-1]]) # 读取标签数据,表示只读取最后一列,[-1]表示最后一行拿出来的是一个矩阵

# 2. 设计模型（用于计算y^而不是计算loss）
# 激活函数还有很多：torch.nn.ReLU()不连续、torch.nn.Tanh()、torch.nn.LeakyReLU()、torch.nn.Softmax()、torch.nn.Softplus()、torch.nn.Softsign()、torch.nn.Softmin()、torch.nn.Softmax2d()、torch.nn.LogSoftmax()等
# 是激活函数和损失函数的选择，各种排列组合试试，哪个效果好就用哪个
class DiabetesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 建立三个模型，每个模型都是一个线性模型，输入是8维，输出是1维
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # 最后构造sigmoid激活函数进行一次非线性变换，将输出值映射到0-1之间
        # 此处用的是torch.nn.Sigmoid()是一个模块类（Module），需要先实例化，然后作为网络的一部分来使用
        # 之前的F.sigmoid()函数是一个函数，不是一个模块，不需要实例化，直接调用即可
        self.sigmoid = torch.nn.Sigmoid()
        self.activate = torch.nn.LeakyReLU() # 可以更换其他激活函数
        
    def forward(self, x):
        # 每一层使用x是因为每一层的输入都是上一层的输出，它代表着通过网络层的逐步处理后的数据流，即在每一步操作后，x 被更新为新值。这是一个很常见的做法
        # x = self.sigmoid(self.linear1(x)) # 第一层线性模型，然后进行sigmoid激活，得到第一层输出O1，作为第二层的输入
        # x = self.sigmoid(self.linear2(x)) # 第二层线性模型，然后进行sigmoid激活，得到第二层输出O2，作为第三层的输入
        # x = self.sigmoid(self.linear3(x)) # 第三层线性模型，然后进行sigmoid激活，得到第三层输出y^，作为最终输出]
        x = self.activate(self.linear1(x)) 
        x = self.activate(self.linear2(x)) 
        x = self.sigmoid(self.linear3(x)) # 若最后一层为relu，可能会导致输出值可能会超出0-1(ReLu)的范围，可以最后改为sigmoid函数
        return x

model = DiabetesModel()
# 3. 构建损失函数loss和优化器optimizer对[-1]象(使用pytorch的API)
criterion = torch.nn.BCELoss(reduction='mean') # 二分类交叉熵损失函数,对每个样本的loss求平均值
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # 实例化一个优化器对象，用于更新权重

epochs = [] # 迭代次数
losses = [] # 损失值

# 4. 训练training cycle（forward、backward、update）
# 注意：如果结果没有收敛，可以增加迭代次数
for epoch in range(1000):
    epochs.append(epoch)

    # (1)前馈过程算y^
    y_pred = model(x_data)# 目前还没有用到mini-batch的方式，先使用所有数据（batch）进行训练.后面讲DataLoader时会使用mini-batch的方式
    loss = criterion(y_pred, y_data) 

    print(epoch, loss.item()) # 打印loss值

    optimizer.zero_grad() # 记住梯度要先清零!

    # (2)反馈过程计算梯度
    loss.backward()
    losses.append(loss.item()) # 记录loss值

    # (3)更新权重（链式法则）
    optimizer.step() 


# 可视化训练结果
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()# 显示网格
plt.show()

