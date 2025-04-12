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
    def __init__(self,file_path):
        # 读取数据，skiprows=1表示跳过第一行，usecols=list(range(1,94))表示只读取第1到93列
        fearures = np.loadtxt(file_path, 
                              delimiter=',',
                              skiprows = 1, 
                              usecols = list(range(1,94)))
        # 读取标签，dtype=str表示标签为字符串，usecols = 94表示只读取第94列
        # 使用pd.get_dummies将标签转换为one-hot编码
        lables_onehot = pd.get_dummies(np.loadtxt(file_path,
                                                  delimiter=',',
                                                  skiprows = 1, 
                                                  dtype=str,
                                                  usecols = 94))
        self.x_data = torch.from_numpy(fearures)
        # 将one-hot编码转换为id表示(0~8)
        self.y_data = torch.from_numpy(np.argmax(lables_onehot.values,axis=1))
        self.len = fearures.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    

train_dataset = OttoDataset("./data/otto/train.csv")
print(train_dataset.y_data[:10])
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=32, 
                          shuffle=True,
                          num_workers=2)    

test_dataset = OttoDataset("./data/otto/test.csv")
test_loader = DataLoader(dataset=test_dataset, 
                          batch_size=32, 
                          shuffle=False,
                          num_workers=2)

for data in test_loader:
    print(data)
    break
