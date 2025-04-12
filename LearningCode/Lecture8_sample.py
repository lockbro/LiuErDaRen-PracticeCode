import numpy as np
import matplotlib.pyplot as plt 
import torch
# Dataset是一个抽象类abstract class，不能实例化，只能被继承并自定义
# 类似一个冰箱，包含了所有数据（食材），但是不能直接使用
from torch.utils.data import Dataset
# DataLoader是一个数据加载器（shuffle、batch_size设置），可以实例化，并对数据进行批量处理
# 类似一个我的厨房助力，可以帮我取出食材（数据），准备好一批批分量供你使用，并且确保这个过程高效、顺畅
from torch.utils.data import DataLoader
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1. 构建数据集
# 糖尿病数据集的类，继承Dataset类
class DiabetesDataset(Dataset):
    # 一般做数据集的初始化，或定义一些读写，根据输出数据类型定义（支持多模态）
    def __init__(self, filepath):
       # 读取数据，并转换为numpy数组存储到xy中
       xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
       # torch.from_numpy(ndarray)函数将numpy数据转换为tensor数据
       # 通用形式： start:end:step
       self.x_data = torch.from_numpy(xy[:,:-1])# 读取特征数据,切片操作，":表示所有行,:-1表示除最后一列之外的所有列",读取除最后一列之外的所有列
       self.y_data = torch.from_numpy(xy[:, [-1]])# 读取标签数据,表示只读取最后一列,[-1]表示最后一行拿出来的是一个矩阵
       # 取矩阵的行数，获取数据集的样本数量（样本数量=行数，特征+标签=列数）
       self.len = xy.shape[0]

    def __len__(self):# 魔法函数，返回数据集的长度(数据条数)
        return self.len

    def __getitem__(self, index):# 魔法方法：实例化后，可以像列表一样，通过索引访问数据样本item
        return self.x_data[index], self.y_data[index] #返回一个元组，包含特征和标签
# 实例化数据集
dataset = DiabetesDataset('./data/diabetes.csv')
# 设置数据加载器
train_loader = DataLoader(dataset=dataset,# 传递数据集
                          batch_size=32,# 设置batch_size  
                          shuffle=True,# 设置是否打乱数据
                          num_workers=2)# 设置多线程(CPU并行读取，提高读取速度)

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
        self.activate = torch.nn.ReLU6() # 可以更换其他激活函数
        
    def forward(self, x):
        # 每一层使用x是因为每一层的输入都是上一层的输出，它代表着通过网络层的逐步处理后的数据流，即在每一步操作后，x 被更新为新值。这是一个很常见的做法
        # x = self.sigmoid(self.linear1(x)) # 第一层线性模型，然后进行sigmoid激活，得到第一层输出O1，作为第二层的输入
        # x = self.sigmoid(self.linear2(x)) # 第二层线性模型，然后进行sigmoid激活，得到第二层输出O2，作为第三层的输入
        # x = self.sigmoid(self.linear3(x)) # 第三层线性模型，然后进行sigmoid激活，得到第三层输出y^，作为最终输出]
        x = self.activate(self.linear1(x)) 
        x = self.activate(self.linear2(x)) 
        x = self.sigmoid(self.linear3(x))
        return x


epochs = [] # 迭代次数
losses = [] # 损失值
training_epochs = 200
model = DiabetesModel()  # 确保在训练循环之前定义模型
# 3. 构建损失函数loss和优化器optimizer对[-1]象(使用pytorch的API)
criterion = torch.nn.BCELoss(reduction='mean') # 二分类交叉熵损失函数,对每个样本的loss求平均值
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 实例化一个优化器对象，用于更新权重

# 需要进行封装，不然会报错
if __name__ == '__main__': 
    # 4. 训练training cycle（forward、backward、update）
    # 注意：如果结果没有收敛，可以增加迭代次数
    for epoch in range(training_epochs):
        epochs.append(epoch)
        total_loss = 0  # 每次迭代开始时，初始化total_loss为0
        for i, data in enumerate(train_loader, 0):  # enumerate主要是为了获得当前迭代的次数
            # (1)获取数据，inputs是特征，labels是标签,都是tensor类型，都是张量
            inputs, labels = data

            # (2)前馈过程算y^
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            print(epoch, i, loss.item())  # 打印loss值

            # (3)反馈过程计算梯度
            optimizer.zero_grad()  # 记住梯度要先清零!
            loss.backward()
            total_loss += loss.item()  # 将loss值累加到total_loss中

            # (4)更新权重（链式法则）
            optimizer.step()

        # 在所有batch结束后，计算每个epoch的平均loss（因为total_loss是所有batch的loss之和）
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)  # 将平均loss添加到losses列表中

    # 可视化训练结果
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()# 显示网格
    plt.show()