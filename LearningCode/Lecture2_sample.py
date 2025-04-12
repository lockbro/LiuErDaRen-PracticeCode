import numpy as np
import matplotlib.pyplot as plt # 该模块主要用于绘制 2D 图表

# 测试集中的坐标点x和y分开存放
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    # 为了得到的结果≥0所以进行了平方
    return (y_pred - y) * (y_pred - y)

# 穷举的所有权重都放在w_list列表中
w_list = []
# 每个权重对应的loss值放在mse_list列表
mse_list = []

# 列举[0.0,4.0]的w值,并计算对应的MSE值
for w in np.arange(0.0, 4.1, 0.1):
    print('w = ', w) # 显示此轮计算的w值
   
    loss_sum = 0 # 每一轮计算将loss值重置
    
    # zip函数可以将分离的x_data,y_data拼成各坐标点:(1,2)、(2,4)、(3,6)
    for x_val, y_val in zip(x_data,y_data):
        y_pred_val = forward(x_val) # 计算预测值
        loss_val = loss(x_val, y_val) # 计算损失值
        loss_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)

    print('MSE = ', loss_sum / 3) # 显示MSE值
    w_list.append(w)
    mse_list.append(loss_sum / 3)

# 可视化结果
plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('MSE_Value')
plt.show()