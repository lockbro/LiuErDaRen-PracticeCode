# 鞍点问题是指在某些情况下，梯度下降会陷入局部最小值,而SGD（Stochastic Gradient Descent，SGD）可以避免这种情况。
# 由于引入的随机性，SGD可以探索更多的参数空间，有可能跳出局部极小值，找到更好的解。
# SGD的更新方向存在噪声，所以它不会严格遵循最陡下降方向，而是趋向于全局最小值的方向，但可能会更慢收敛到一个接近最优的点。

# 与普通的梯度下降不同，之前的梯度下降是对所有样本进行计算，SGD是随机的，每次训练只使用一个样本而SGD是随机对单个样本进行计算
# 由于SGD是随机的，所以每次训练的结果可能不同，但是通常会比普通梯度下降更快

import numpy as np
import matplotlib.pyplot as plt # 该模块主要用于绘制 2D 图表
import random

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# w为权重，设置初始值为1.0
w = 1.0

# 前向传播函数，返回预测值
def forward(x):
    return x * w

# 为了和普通梯度下降比较，这里将损失函数改为loss
# 输入参数不再是数据集x_data和y_data，而是单个样本
def loss(x, y):
    y_pred = forward(x) # 对单个样本进行预测
    # 返回平均损失值（除以样本数量）
    return (y_pred - y) ** 2

# 梯度函数（对loss的w求偏导），返回梯度值
# 不再对所有样本进行计算，而是对单个样本进行计算
def gradient(x, y):
    return 2 * x * (x * w - y)  

print('Predict (before training)', 4, forward(4))

epochs = [] # 迭代次数
losses = [] # 损失值
α = 0.01 # 学习率
# SGD训练过程
for epoch in range(100):
    index = random.randint(0,2) # 随机抽取数据集中的样本
    x = x_data[index]
    y = y_data[index]
    grad_val = gradient(x, y)# 计算单个样本的梯度值
    w -= α * grad_val # 更新权重
    # 显示训练过程,结果精确到小数点后2位
    print('\tgrad:',x,y,round(grad_val,2))
    

    loss_val = loss(x, y)# 计算单个样本的损失值
    epochs.append(epoch)
    losses.append(loss_val)
    print('Epoch:', epoch, 'w=', round(w, 2), 'loss=', round(loss_val, 2))

print('Predict (after training)', 4, round(forward(4), 2))

# 可视化结果
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.grid()# 显示网格
plt.show()

# 在对所有数据计算普通的梯度下降的时候，各个点的计算是独立的，所以可以并行计算
# 而在SGD的计算中，各个点的计算是串行的，因为每次只计算一个点，下一个点的计算需要等待上一个点的w权重计算的结果，所以不能并行计算
# 但是由于SGD的计算是随机的，所以可以通过并行计算多个SGD的计算过程来提高计算速度

# 因此为了折中，通常会使用小批量随机梯度下降（Mini-batch Stochastic Gradient Descent，Mini-batch SGD）
# 这是是介于普通梯度下降和SGD之间的一种方法，它每次计算一小部分样本的梯度，然后更新权重
# 通过调整小批量的大小，可以在计算速度和计算精度之间取得平衡
# 现在很多接口的Batch指的都是Mini-batch，而不是普通的梯度下降
# 在深度学习中，通常也都使用Mini-batch SGD来训练模型