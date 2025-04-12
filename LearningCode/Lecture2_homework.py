import numpy as np

# 该模块主要用于绘制 2D 图表
import matplotlib.pyplot as plt 

# 该模块主要用于绘制 3D 图表
from mpl_toolkits.mplot3d import Axes3D 

# 测试集中的坐标点x和y分开存放
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    # 为了得到的结果≥0所以进行了平方
    return (y_pred - y) ** 2

# 每个权重对应的loss值放在mse_list列表
mse_list = []


# [X,Y]=np.meshgrid(x,y) 
# 将向量x和y两个坐标轴上的点，转化为平面上的“网格”
# （即将两组一维数据合成为一个二维数据）
w_list = np.arange(0.0, 4.1, 0.1)
b_list = np.arange(-2.0, 2.1, 0.1)
ww, bb = np.meshgrid(w_list, b_list)

for w in w_list:
    for b in b_list:    
        # 显示当前参数状态
        print('w = {0}, b = {0}'.format(w, b))
        
        # 每一轮计算将loss值重置
        loss_sum = 0 
        
        # zip函数可以将分离的x_data,y_data拼成各坐标点
        for x_val, y_val in zip(x_data,y_data):
            y_pred_val = forward(x_val) # 计算预测值
            loss_val = loss(x_val, y_val) # 计算损失值
            loss_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)

        print('MSE = ', loss_sum / 3) # 显示MSE值
        mse_list.append(loss_sum / 3)
        # print(mse_list)

# 使用arrary根据w_list和b_list的维度和mse_list的结果构建2维数组mse
mse = np.array(mse_list).reshape(w_list.shape[0],b_list.shape[0])
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
# X, Y, Z都必须是2D arrays
surf = ax.plot_surface(ww, bb, mse, rstride=1, cstride=1, cmap='rainbow')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()