# RNN(Recurrent Neural Network)循环神经网络
# 专门用于处理序列数据（时间序列、文本序列等），具有记忆能力，能够捕捉序列中的时序关系
# 适用于自然语言处理、语音识别、时间序列预测等任务
# 权重共享：RNN在每个时间步使用相同的权重矩阵进行计算，这使得它能够处理任意长度的序列数据，减少了参数的数量
# 但是RNN在长序列中容易出现梯度消失或梯度爆炸的问题，导致模型难以学习长期依赖关系
# 本质是个线性层，但是在每个时间步上都使用相同的权重矩阵进行计算

# 如果超参数假设：
    # batch_size = 1 # 相当于几句话
    # sequence_length = 3  # 序列长度(包含几个序列):x1,x2,x3，相当于每句话有几个单词
    # input_size = 4  # 每个序列包含的特征数（元素个数），相当于每个单词有几个特征（词嵌入向量维度）
    # hidden_size = 2 # 隐藏层中每个块包含的元素个数
# 那么RNNCell的输入和输出的维度分别是：
    # 输入：input.shape = batch_size * input_size = 1 * 4 = 4
    # 输出：output.shape = batch_size * hidden_size = 1 * 2 = 2 (因为隐藏层中每个的输入是下一个单词的输出，因此维度是一样的)
# 因此，整个的序列将被构造成一个张量：
    # data.shape = (sequence_length, batch_size, input_size) = (3, 1, 4) = (3, 4)

# 最重要是搞清楚维度的变化

# 训练个模型用于训练“hello” -> "ohlol"(Seq2Seq,即将“hello”中的每个字母向后移动一个位置)
# 只使用RNNCell实现
import torch
import torch.nn as nn
import torch.optim as optim
batch_size = 1 # 相当于几句话
sequence_length = 5  # 五个字母
input_size = 4  # 每个字母转换为索引后再转为one-hot向量，那么每个字母有4个特征（one-hot向量维度）
hidden_size = 4 # 隐藏层中每个块包含的元素个数（h、e、l、o）

# 1.准备数据
idx2char = ['e', 'h', 'l', 'o'] # 字典，索引到字母的映射,用于之后便于打印输出结果
x_data = [1, 0, 2, 2, 3] # hello的索引表示
y_data = [3, 1, 2, 3, 2] # ohlol的索引表示

# 生成用于查询的one-hot向量表，shape=(4, 4)
ont_hot_lookup = torch.eye(len(idx2char)) 
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])

# hello的one-hot向量表示，shape=(5, 4)
x_one_hot = [ont_hot_lookup[x] for x in x_data]

# 将one-hot向量列表转换为张量，使用view()函数将其转换为指定的形状，-1表示自动计算序列长度seqLen
inputs = torch.stack(x_one_hot).view(-1, batch_size, input_size) # 转换为张量，shape=(5, 1, 4)
# tensor([[[0., 1., 0., 0.]],

#         [[1., 0., 0., 0.]],

#         [[0., 0., 1., 0.]],

#         [[0., 0., 1., 0.]],

#         [[0., 0., 0., 1.]]])

# 将标签转为(seqLen, 1)的张量,-1表示自动计算序列长度
labels = torch.LongTensor(y_data).view(-1, 1)

# 2. 定义模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 定义RNNCell
        self.rnncell = nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)
    
    # 前向传播,将输入和隐藏状态作为输入，返回新的隐藏状态
    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden) # 只有一个隐藏层
        return hidden

    # 初始化隐藏状态h_0，batch_size在这里才有用
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)

# 3. 实例化模型
net = Model(input_size, hidden_size, batch_size)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)

# 5. 训练模型
for epoch in range(15):
    # 初始化隐藏状态
    loss = 0
    optimizer.zero_grad() # 梯度清零
    hidden = net.init_hidden() # 初始化隐藏状态
    print("Predicted string: ", end='')
    for input, label in zip(inputs, labels):
        # 前向传播
        hidden = net(input, hidden)
        # 计算损失
        loss += criterion(hidden, label) # 还不用item()，因为还需要构造计算图把loss相加
        # 打印输出
        _, idx = hidden.max(dim=1) # 找到概率最大的索引
        print(idx2char[idx.item()], end='')# 每一轮训练打印一个预测的字符串结果
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    print(", Epoch [%d/15] loss=%.4f" % (epoch+1, loss.item()))

