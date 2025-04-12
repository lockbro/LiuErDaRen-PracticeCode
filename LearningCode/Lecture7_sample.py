# 使用pytorch处理多维特征输入的Logistic Regression，并测试新的数据

# 每一列称之为一个特征（或者字段），每一行称之为一个样本

import numpy as np
import matplotlib.pyplot as plt # 该模块主要用于绘制 2D 图表
import torch
import torch.nn.functional as F # 该模块包含了很多激活函数和损失函数
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1. 构建数据集
# 使用mini-batch的方式,一次性输入多个数据并得到多个预测值
# 因此x_data和y_data都必须是矩阵
# 该例子是单维度，因此每一条数据只有一个维度
x_data = torch.tensor([[1.0], [2.0], [3.0]],requires_grad=True) # 3*1的矩阵
# tensor([[1.0],
#         [2.0],
#         [3.0]])
# 因为此处是二分类问题，因此y_data的取值只能是0或1
y_data = torch.tensor([[0.0], [0.0], [1.0]],requires_grad=True) # 3*1的矩阵,表示对应上面三个样本的分类结果
# tensor([[0.0], 
#         [0.0], 
#         [1.0]])

# 2. 设计模型（用于计算y^而不是计算loss）
# 将z = wx+b称之为一个Affine Unit(仿射单元，有偏置项)
# 将z = wx称之为Linear Unit(线性单元，无偏置项)
# 模型类必须继承torch.nn.Module模块，因为里面有很多模型的基本方法
class Model(torch.nn.Module):
    def __init__(self): # 构造函数，由于不涉及参数，因此和之前的LinearModel基本一样
        # 调用父类的构造函数
        super(Model, self).__init__() 
        # torch.nn.Linear是一个类，用于构建线性模型对象，可以完成对z的计算
        # 输入和输出样本的维度都是1，bias=True表示有偏置项,默认为True
        self.linear = torch.nn.Linear(8, 1) 
        # 神经网络的本质就是寻找一种非线性的空间变换函数
        # 将输入数据从其原生空间变换到一个新的空间，在这个新的空间中，任务（如分类、回归等）变得更为简单和直观
        # linear layer部分就是σ(X*W+b),
        # 其中X是输入的特征数量也就是列数（列的数量表示features），因为为N*8，所以是8个特征
        # W是权重，维度为8*1
        # b是偏置项,维度通过广播机制自动匹配
        # 最后输出的特征数量是1,因为是二分类问题，N*1
        self.sigmoid = torch.nn.Sigmoid() # 定义一个激活函数对象，用于将输出值映射到0-1之间

    # 要覆盖父类的forward函数
    def forward(self, x):# 前向传播函数，实际上执行的是wx+b的操作
        # sigmoid函数用于将输出值映射到0-1之间
        y_pred = self.sigmoid(x)
        return y_pred
    
    # 此处不需要定义backward函数，因为pytorch会自动完成反向传播
    # 但是需要定义forward函数，因为pytorch不知道如何计算y^

# 实例化模型,model是一个callable的对象
model = Model()


# 3. 构建损失函数loss和优化器optimizer对象(使用pytorch的API)
# 实例化一个损失函数对象，用于计算loss
# 使用二分类交叉熵损失函数，用于反映模型的输出和真实输出分布之间的差异
criterion = torch.nn.BCELoss(reduction='sum')
# reduction='sum'表示对每个样本的loss求和,即不对每个样本的loss求平均值（影响学习率的选择）

# 实例化一个优化器对象，用于更新权重
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# model.parameters()递归地返回model中的所有成员参数
# lr是学习率learning rate，设置为1以快速收敛

epochs = [] # 迭代次数
losses = [] # 损失值

# 4. 训练training cycle（forward、backward、update）
# 注意：如果结果没有收敛，可以增加迭代次数
for epoch in range(4000):
    epochs.append(epoch)

    # (1)前馈过程算y^
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data) 

    print(epoch, loss.item()) # 打印loss值

    optimizer.zero_grad() # 记住梯度要先清零!

    # (2)反馈过程计算梯度
    loss.backward()
    losses.append(loss.item()) # 记录loss值

    # (3)更新权重（链式法则）
    optimizer.step() 


# 打印训练后的权重和偏置项
# model.linear.weight是一个tensor对象，使用item()方法可以取出tensor的数值部分
# print('\n')
# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())

# 可视化训练结果
# plt.plot(epochs, losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid()# 显示网格
# plt.show()

# 测试之前训练好的模型的性能
X = np.linspace(0, 10, 200) # 生成0-10之间的200个等差数列
x_test = torch.tensor(X).float().view(200, 1) # 200*1的矩阵
# view(200, 1)表示将200个数据转换为200*1的矩阵,相当于reshape
y_test = model(x_test) # 调用训练好的模型进行测试，返回的是一个200*1的张量
# y_test是一个概率值，表示通过模型预测的概率值
# 如果y_test>0.5,则表示预测为1，否则预测为0
Y = y_test.data.numpy() # 将tensor对象转换为numpy对象（即数组）,方便后续绘图

# 可视化测试结果
plt.plot(X, Y)
plt.plot([0, 10], [0.5, 0.5], c='r') # 绘制y=0.5的虚线
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()# 显示网格
plt.show()
