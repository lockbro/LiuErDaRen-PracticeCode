# https://medium.com/@akashgajjar8/titanic-survival-prediction-using-pytorch-a5b9fb4eca53
# 使用dataloader方法实现Titanic生存预测

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

train_dataset = pd.read_csv('./data/titanic/train.csv')
test_dataset = pd.read_csv('./data/titanic/test.csv')

class TitanicDataset(Dataset):
    def __init__(self, file_name, Train=True):
        # 读取数据，并转换为numpy数组存储到xy中
        self.dataframe = pd.read_csv(file_name)
        # 删除Age为null的行 
        self.dataframe = self.dataframe[~self.dataframe['Age'].isnull()]
        # 重置索引
        self.dataframe = self.dataframe.reset_index()
        # 如果Train为True，则删除Survived列
        self.Train = Train

    def __len__(self):
        # 获取数据集的样本数量
        return self.dataframe.shape[0]

    # 获取数据集的特征和标签
    def __getitem__(self, idx):
        # 如果idx是tensor，则转换为列表
        if(torch.is_tensor(idx)):
            idx = idx.tolist();
        
        # 如果Train为True，则获取标签（因为训练集有标签，测试集没有标签）
        if(self.Train): 
            # 获取标签
            survived = self.dataframe['Survived']
            # 将标签转换为numpy数组
            survived = np.array(survived)[idx]
        
        # 获取特征
        # 创建一个DataFrame，用于存储特征
        # 特征：Sex, Pclass, Age
        features = pd.DataFrame(columns=('Sex', 'Pclass', 'Age'))
        # 将特征转换为numpy数组,idx是索引，loc[idx]是索引对应的行
        features.loc[idx] = [ 1 if self.dataframe.loc[idx]['Sex']=='male' else 0, \
                              self.dataframe.loc[idx,'Pclass'],\
                              self.dataframe.loc[idx,'Age']]
        # 将特征转换为numpy数组
        features = np.array(features)
        
        # 如果Train为True，则返回特征和标签（训练集 特征+标签）
        if(self.Train):
            sample = (features, survived)
        # 如果Train为False，则返回特征（测试集 特征）
        else:
            sample = features
        return sample

train_dataset = TitanicDataset('./data/titanic/train.csv', Train=True)
test_dataset = TitanicDataset('./data/titanic/test.csv', Train=False)
train_dataloader = DataLoader(dataset=train_dataset, 
                              batch_size=100, 
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, 
                             batch_size=100, 
                             shuffle=False)

# 2. 设计模型（用于计算y^而不是计算loss）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 建立三个模型，每个模型都是一个线性模型，输入是3维，输出是1维
        self.linear1 = torch.nn.Linear(3, 4)
        self.linear2 = torch.nn.Linear(4, 5)
        self.linear3 = torch.nn.Linear(5, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.activate = torch.nn.ReLU() # 可以更换其他激活函数
        
    def forward(self, x):
        x = self.activate(self.linear1(x)) 
        x = self.activate(self.linear2(x)) 
        x = self.sigmoid(self.linear3(x))
        return x
    
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs):
    # 用于存储训练过程中的指标
    train_losses = []
    train_accuracies = []
    epochs = []

    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练过程
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.view(-1, 3).float()
            labels = labels.view(-1, 1).float()
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct / total
        
        # 存储训练指标
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        epochs.append(epoch + 1)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    print("训练完成！")
    
    # 绘制损失下降曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Decrease Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return epochs, train_losses, train_accuracies

def evaluate_model(model, test_dataloader, criterion):
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    
    print("\n开始测试...")
    with torch.no_grad():
        for data in test_dataloader:
            # For test data, we only have inputs (no labels)
            inputs = data  # Get only the inputs
            inputs = inputs.view(-1, 3).float()
            
            # Get model predictions
            outputs = model(inputs)
            
            # Since we don't have actual labels for test data,
            # we'll just calculate the predictions
            predictions = (outputs > 0.5).float()
            total += inputs.size(0)
            
            # 计算准确率
            # 这里需要确保labels存在于data中
            if hasattr(data, 'labels'):
                labels = data.labels.view(-1, 1).float()  # Assuming labels are available in the data
                correct += (predictions == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    print(f'测试完成 - 总样本数: {total}, 准确率: {accuracy:.4f}')
    return predictions

def plot_test_predictions(predictions):
    plt.figure(figsize=(10, 5))
    
    # Plot predictions distribution
    plt.subplot(1, 2, 1)
    plt.hist(predictions.numpy(), bins=2, rwidth=0.8)
    plt.title('Distribution of Predictions')
    plt.xlabel('Prediction (0: Not Survived, 1: Survived)')
    plt.ylabel('Count')
    
    # Plot prediction percentages
    plt.subplot(1, 2, 2)
    survived = (predictions == 1).sum().item()
    not_survived = (predictions == 0).sum().item()
    plt.pie([survived, not_survived], 
            labels=['Survived', 'Not Survived'],
            autopct='%1.1f%%')
    plt.title('Prediction Distribution')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == '__main__':
    # 模型参数设置
    model = Model()
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 降低学习率以减少震荡
    num_epochs = 200  # 增加训练轮数以提高模型稳定性

    # 训练模型
    epochs, train_losses, train_accuracies = train_model(
        model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs
    )

    # 获取测试集预测结果
    predictions = evaluate_model(model, test_dataloader, criterion)

    # 可视化预测结果
    plot_test_predictions(predictions)