# 使用pytorch实现线性回归,使用不同的优化器来对比结果
# 重点是构造计算图而再是计算梯度，因为构造了计算图，pytorch会自动计算梯度

import matplotlib.pyplot as plt # 该模块主要用于绘制 2D 图表
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 1. 构建数据集
# 使用mini-batch的方式,一次性输入多个数据并得到多个预测值
# 因此x_data和y_data都必须是矩阵
x_data = torch.tensor([[1.0], [2.0], [3.0]]) # 3*1的矩阵
# tensor([[1.],
#         [2.],
#         [3.]])
y_data = torch.tensor([[2.0], [4.0], [6.0]]) # 3*1的矩阵
# tensor([[2.],
#         [4.],
#         [6.]])

# 2. 设计模型（用于计算y^而不是计算loss）
# 将z = wx+b称之为一个Affine Unit(仿射单元，有偏置项)
# 将z = wx称之为Linear Unit(线性单元，无偏置项)
# 模型类必须继承torch.nn.Module模块，因为里面有很多模型的基本方法
class LinearModel(torch.nn.Module):
    def __init__(self): # 构造函数,初始化对象的参数
        # 调用父类的构造函数
        super(LinearModel, self).__init__() 
        # torch.nn.Linear是一个类，用于构建线性模型对象，可以完成对z的计算
        # 输入和输出样本的维度都是1，bias=True表示有偏置项,默认为True
        self.linear = torch.nn.Linear(1, 1)
    
    # 要覆盖父类的forward函数
    def forward(self, x):# 前向传播函数，实际上执行的是wx+b的操作
        y_pred = self.linear(x) # 可以直接调用torch.nn.Linear的__call__方法
        return y_pred
    
    # 此处不需要定义backward函数，因为pytorch会自动完成反向传播
    # 但是需要定义forward函数，因为pytorch不知道如何计算y^

# 实例化模型,model是一个callable的对象
model_SGD = LinearModel()
model_Adagrad = LinearModel()
model_Adam = LinearModel()
model_Adamax = LinearModel()
model_ASGD = LinearModel()
# model_LBFGS = LinearModel()
model_RMSprop = LinearModel()
model_Rprop = LinearModel()
models = [model_SGD, model_Adagrad, model_Adam, model_Adamax, model_ASGD, model_RMSprop, model_Rprop]

# 3. 构建损失函数loss和优化器optimizer对象(使用pytorch的API)
# 实例化一个损失函数对象，用于计算loss
# criterion = torch.nn.MSELoss(size_average=False)
# MSELoss继承自torch.nn.Module
# size_average=False表示不对每个样本的loss求平均值
criterion = torch.nn.MSELoss(reduction='sum')
# reduction='sum'表示对每个样本的loss求和,即不对每个样本的loss求平均值

# 实例化一个优化器对象，用于更新权重
opt_SGD = torch.optim.SGD(model_SGD.parameters(), lr=0.01)
opt_Adagrad = torch.optim.Adagrad(model_Adagrad.parameters(), lr=0.01)
opt_Adam = torch.optim.Adam(model_Adam.parameters(), lr=0.01)
opt_Adamax = torch.optim.Adamax(model_Adamax.parameters(), lr=0.01)
opt_ASGD = torch.optim.ASGD(model_ASGD.parameters(), lr=0.01)
# opt_LBFGS = torch.optim.LBFGS(model_LBFGS.parameters(), lr=0.01)
opt_RMSprop = torch.optim.RMSprop(model_RMSprop.parameters(), lr=0.01)
opt_Rprop = torch.optim.Rprop(model_Rprop.parameters(), lr=0.01)
opts = [opt_SGD, opt_Adagrad, opt_Adam, opt_Adamax, opt_ASGD, opt_RMSprop, opt_Rprop]
# model.parameters()递归地返回model中的所有成员参数
# lr是学习率learning rate

Fig_title = ['SGD','Adagrad','Adam','Adamax','ASGD','RMSprop','Rprop']
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
index = 0
# 4. 训练training cycle（forward、backward、update）
# 注意：如果结果没有收敛，可以增加迭代次数
for item in zip(models, opts):
    epochs = [] # 迭代次数
    losses = [] # 损失值

    model = item[0]
    optimizer = item[1]
    
    for epoch in range(500):
        epochs.append(epoch)

    # (1)前馈过程算y^
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data) 

        print(epoch, loss.item()) # 打印loss值

        optimizer.zero_grad() # 反向传播前记住梯度清零!

    # (2)反馈过程计算梯度
        loss.backward()
        losses.append(loss.item()) # 记录loss值

    # (3)更新权重
        optimizer.step()

    # 打印训练后的权重和偏置项
    # model.linear.weight是一个tensor对象，使用item()方法可以取出tensor的数值部分
    print('\n')
    print('w=', model.linear.weight.item())
    print('b=', model.linear.bias.item())

    # 测试模型，假如输入x=4.0，预测y^
    x_test = torch.tensor([[4.0]]) # 1*1的矩阵
    y_test = model(x_test) # 返回的是一个1*1的矩阵
    print('y_pred=', y_test.item())


    # 可视化结果
    a = int(index / 4) # 行
    b = int(index % 4)  # 列
    axs[a, b].plot(epochs, losses)
    axs[a, b].set_title(Fig_title[index])
    axs[a, b].set_xlabel('Epoch')
    axs[a, b].set_ylabel('Loss') 
    index += 1
plt.show()