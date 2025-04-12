import numpy as np
import matplotlib.pyplot as plt 
import torch
import pandas as pd
import torchvision
# Dataset是一个抽象类abstract class，不能实例化，只能被继承并自定义
# 类似一个冰箱，包含了所有数据（食材），但是不能直接使用
# DataLoader是一个数据加载器（shuffle、batch_size设置），可以实例化，并对数据进行批量处理
# 类似一个我的厨房助力，可以帮我取出食材（数据），准备好一批批分量供你使用，并且确保这个过程高效、顺畅
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F # 激活函数,引入relu
import torch.optim as optim # 优化器
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1. 构建数据集
class OttoDataset(Dataset):
    def __init__(self, file_path):
        # 使用pandas读取CSV文件
        df = pd.read_csv(file_path)
        
        # 分离特征和标签
        if 'target' in df.columns:  # 训练集
            self.features = torch.FloatTensor(df.drop(['id', 'target'], axis=1).values)
            labels = df['target']
            self.labels = pd.get_dummies(labels).values
        else:  # 测试集
            self.features = torch.FloatTensor(df.drop(['id'], axis=1).values)
            self.labels = None  # 测试集没有标签
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], torch.FloatTensor(self.labels[idx])
        return self.features[idx]
    

train_dataset = OttoDataset("./data/otto/train.csv")
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=32, 
                          shuffle=True,
                          num_workers=2)        
test_data = pd.read_csv("./data/otto/test.csv")
X_test = torch.FloatTensor(test_data.drop(['id'], axis=1).values)

# 2. 设计模型（用于计算y^而不是计算loss）
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入特征维度93,因为排除id和target，剩下93个特征
        self.linear1 = torch.nn.Linear(93, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.linear4 = torch.nn.Linear(16, 9)# 输出特征维度9(0~8)

    def forward(self, x):
        # 第一步：展平,展平后x的形状为(batch_size, 93),-1表示自动计算
        x = x.view(-1, 93)
        # 第二步：激活函数
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # 第三步：输出，最后一层不做激活，因为之后马上要进行softmax
        return self.linear4(x)

model = Net()  # 确保在训练循环之前定义模型

# 3. 构建损失函数loss和优化器optimizer对象(使用pytorch的API)
criterion = torch.nn.CrossEntropyLoss()  
# 因为网络模型有点大，所以最好带动量避免局部最优
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

# 4. 训练,封装一轮训练函数
def train(epoch):
    training_losses = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
        # (1)获取数据，inputs是特征，targets是分类标签,都是tensor类型，都是张量
        # 由于循环中已经用元组表示，所以不需要再解包

        # 一定记得在前馈计算之前梯度清零，不然会累加
        optimizer.zero_grad()
        
        # (2)前馈过程算y^
        outputs = model(inputs)
        # (3)计算loss
        loss = criterion(outputs, targets)
        # (4)计算梯度
        loss.backward()
        # (5)更新权重
        optimizer.step()

        # 累加loss
        training_losses += loss.item() # 使用item()将loss转换为标量，不然会构建计算图

        # 每500个iterations打印一次loss,不然输出太多计算成本增加
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, training_losses / 300))
            training_losses = 0.0

# 封装测试函数,只需算正向传播，不需要计算梯度（torch.no_grad）
def test():
    model.eval()
    with torch.no_grad():# 下面的代码不会再计算梯度
        test_outputs = model(X_test)
        test_probabilities = F.softmax(test_outputs, dim=1).cpu().numpy()
    predictions_df = pd.DataFrame(test_probabilities, columns=['Class_1', 
                                                               'Class_2', 
                                                               'Class_3', 
                                                               'Class_4', 
                                                               'Class_5', 
                                                               'Class_6', 
                                                               'Class_7', 
                                                               'Class_8', 
                                                               'Class_9'])
    # 插入id列, 放在最前面
    predictions_df.insert(0, 'id', test_data['id'])
    # 保存预测结果,index=False表示不保存行索引
    predictions_df.to_csv('predictions.csv', index=False)

training_epochs = 10
# 需要进行封装，不然会报错
if __name__ == '__main__': 
    # 训练
    for epoch in range(training_epochs):
        train(epoch)
    # 测试
    test()
