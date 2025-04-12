import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# w为权重，设置初始值为1.0（为tensor），且需要求导（计算梯度）
w = torch.tensor([1.0], requires_grad=True)

def forward(x):
    # 由于w是tensor，所以这里的乘法已经被重载，是tensor之间的乘法（x自动被转为tensor）
    # 返回的结果也是tensor，且需要计算梯度
    return x * w 

# 损失函数，返回损失值
# （构建新的计算图）
def loss(x, y): 
    y_pred = forward(x)
    return (y_pred - y) ** 2

epochs = [] # 迭代次数
losses = [] # 损失值

# 训练过程
print('Predict (before training)', 4, round(forward(4).item(), 2))

for epoch in range(100):
    epochs.append(epoch)
    for x, y in zip(x_data, y_data):
        # 前馈运算，计算损失
        l = loss(x, y) 

        # 反馈运算，计算梯度
        # tensor数据类型的backward()方法用于计算梯度,并将结果存在w.grad这个tensor中
        # （完成反向计算后，此轮的loss计算图自动释放）
        l.backward() 

        # 显示训练过程
        # .item()方法用于取tensor的数值部分，并转为一个标量
        print('\tgrad:', x, y, round(w.grad.item(), 2))

        # 更新权重（单纯修改数值，因此需要取.data）
        # 由于w.grad也是tensor类型，所以还需要取w.grad的data属性（tensor的数据部分）以便进行数值运算
        w.data -= 0.01 * w.grad.data

        # 需要手动清空梯度（否则梯度会累加）
        w.grad.data.zero_()
    losses.append(l.item())
    # 显示训练过程
    print('Epoch:', epoch, 'w=', round(w.item(), 2), 'loss=', round(l.item(), 2))

print('Predict (after training)', 4, round(forward(4).item(), 2))

# 可视化结果
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()# 显示网格
plt.show()