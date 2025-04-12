# 独热向量的缺点：
# 1. 稀疏性sparse：每个向量都是稀疏的，只有一个元素为1，其他元素都为0，这导致了存储空间的浪费
# 2. 高维度high-demension：每个向量的维度等于词汇表的大小，这使得计算量和存储量都很大
# 3. hardcode：独热向量是硬编码的，无法表示不同的语义

# 因此，需要使用其他的方法来表示文本数据，如embedding vector,有如下优点：
# 1. 低维度low-demension：每个向量的维度小于词汇表的大小，这使得计算量和存储量都小
# 2. 可训练trainable：embedding vector是可训练的，这使得它可以表示不同的语义
# 3. 可以表示不同的语义：embedding vector可以表示不同的语义，这使得它可以表示不同的语义

# 训练个模型用于训练“hello” -> "ohlol"(Seq2Seq,即将“hello”中的每个字母向后移动一个位置)
# 使用完整的RNN实现,加入embedding层和线性层
# 线性层的作用是调整输出维度，可以升维也可以降维，输出最终符合我们需求的维度
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
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = nn.Embedding(input_size, embedding_size) # 定义embedding层
        # 定义RNN
        # RNN的输入和输出的形状是(seqLen, batch_size, input_size)和(seqLen, batch_size, hidden_size)
        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True) # 调整维度顺序在前
        # 定义线性层，将RNN的输出转换为与Labels的维度一致
        self.fc = nn.Linear(hidden_size, num_classes) # 定义线性层

    # 前向传播,将输入和隐藏状态作为输入，返回新的隐藏状态
    # 不需要在参数上引入batch_size，因为这里已经在构造隐藏状态h_0时使用了batch_size
    def forward(self, x):
        # 初始化隐藏状态h_0
        hidden = torch.zeros(num_layers, x.size(0), hidden_size) # 初始化隐藏状态h_0
        x = self.emb(x) # 将输入转换为嵌入向量
        x, _ = self.rnn(x, hidden) # 前向传播
        x = self.fc(x) # 将RNN的输出转换为与Labels的维度一致
        return x.view(-1, num_classes) # 将输出转换为与Labels的维度一致

# 3. 实例化模型
net = Model()

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.05)

# 5. 训练模型,使用RNN
for epoch in range(15):
    optimizer.zero_grad() # 梯度清零
    outputs = net(inputs)
    # 计算损失的outputs和labels的维度是(seqLen*batch_size, hidden_size)和(seqLen*batch_size)
    loss = criterion(outputs, labels) 
    loss.backward() # 反向传播
    optimizer.step() # 更新参数

    # 打印结果
    _, idx = outputs.max(dim=1) # 找到每个时间步上输出概率最大的索引
    idx = idx.data.numpy() # 转换为numpy数组
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


