import numpy as np
import matplotlib.pyplot as plt # 该模块主要用于绘制 2D 图表

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# w为权重，设置初始值为1.0
w = 1.0

# 前向传播函数，返回预测值
def forward(x):
    return x * w

# 损失（目标）函数，返回损失值
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    # 返回平均损失值（除以样本数量）
    return cost / len(xs)

# 梯度函数（对cost的w求偏导），返回梯度值
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    # 返回平均    
    return grad / len(xs)

print('Predict (before training)', 4, forward(4))

epochs = [] # 迭代次数
costs = [] # 损失值
α = 0.01 # 学习率
# 普通的梯度下降（gradient Descent）训练过程
for epoch in range(100):
    epochs.append(epoch)
    cost_val = cost(x_data, y_data)
    costs.append(cost_val)
    grad_val = gradient(x_data, y_data)
    w -= α * grad_val # 更新权重
    # 显示训练过程,结果精确到小数点后2位

    print('Epoch:', epoch, 'w=', round(w, 2), 'loss=', round(cost_val, 2))

print('Predict (after training)', 4, round(forward(4), 2))

# 可视化结果
plt.plot(epochs, costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid()# 显示网格
plt.show()
# 从结果可以看出，随着迭代次数的增加，损失值逐渐减小
# 然而真实情况基本不会平稳下降（会局部震荡），因为数据集中的噪声和样本数量等因素会影响训练结果
# 可以引入指数加权平均（Exponential Moving Average）来减小震荡
# 也可以引入动量（Momentum）来减小震荡