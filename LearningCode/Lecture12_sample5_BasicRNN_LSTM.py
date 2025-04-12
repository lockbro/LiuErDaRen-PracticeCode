# 独热向量的缺点：
# 1. 稀疏性sparse：每个向量都是稀疏的，只有一个元素为1，其他元素都为0，这导致了存储空间的浪费
# 2. 高维度high-demension：每个向量的维度等于词汇表的大小，这使得计算量和存储量都很大
# 3. hardcode：独热向量是硬编码的，无法表示不同的语义

# 因此，需要使用其他的方法来表示文本数据，如embedding vector,有如下优点：
# 1. 低维度low-demension：每个向量的维度小于词汇表的大小，这使得计算量和存储量都小
# 2. 可训练trainable：embedding vector是可训练的，这使得它可以表示不同的语义
# 3. 可以表示不同的语义：embedding vector可以表示不同的语义，这使得它可以表示不同的语义

# 训练个模型用于训练“hello” -> "ohlol"(Seq2Seq,即将“hello”中的每个字母向后移动一个位置)
# 使用LSTM实现
import torch
import torch.nn as nn
import torch.optim as optim
batch_size = 1 # 相当于几句话
sequence_length = 5  # 五个字母
input_size = 4  # 每个字母转换为索引后再转为one-hot向量，那么每个字母有4个特征（one-hot向量维度）
hidden_size = 4 # 隐藏层中每个块包含的元素个数（h、e、l、o）
embedding_size = 10 # embedding vector的维度
num_layers = 2 # RNN的层数
num_classes = 4 # 输出的类别数

# 1.准备数据
idx2char = ['e', 'h', 'l', 'o'] # 字典，索引到字母的映射,用于之后便于打印输出结果
x_data = [[1, 0, 2, 2, 3]] # hello的索引表示,这里是二维的，因为RNN的输入是二维的
y_data = [3, 1, 2, 3, 2] # ohlol的索引表示
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

# 2. 定义模型
# 修改超参数
embedding_size = 4    # 与输入维度相同
hidden_size = 4       # 与输入维度相同
num_layers = 1        # 使用单层LSTM

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 增加了一个h,用于记录长期记忆
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size)
        # 增加了一个c,用于记录短期记忆
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size) 
        
        x = self.emb(x)
        x, _ = self.lstm(x, (h_0, c_0))
        x = self.fc(x)
        return x.view(-1, num_classes)

# 3. 实例化模型
net = Model()

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 修改优化器和训练参数
optimizer = optim.Adam(net.parameters(), lr=0.1)  # 增大学习率
num_epochs = 100  # 增加训练轮数

# 修改训练循环
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 每10轮打印一次结果
    if (epoch + 1) % 10 == 0:
        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/%d] loss = %.3f' % (epoch + 1, num_epochs, loss.item()))