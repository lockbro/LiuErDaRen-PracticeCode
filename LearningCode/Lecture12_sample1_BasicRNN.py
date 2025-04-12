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

import torch

batch_size = 1 # 相当于几句话
sequence_length = 3  # 序列长度(包含几个序列):x1,x2,x3，相当于每句话有几个单词
input_size = 4  # 每个序列包含的特征数（元素个数），相当于每个单词有几个特征（词嵌入向量维度）
hidden_size = 2 # 隐藏层中每个块包含的元素个数

# 构造RNNCell，仅仅仅是一个单元格，表示 RNN 的单个时间步（step）的计算单元
# 不能处理序列数据
cell_1 = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size) # 定义RNNCell

# 构造完整RNN模型，包含多个时间步的计算单元
num_layer = 1 # RNN的层数，默认是1层,多了的话计算很耗时间
# batch_first=False（默认情况）：表示输入和输出数据的形状为 (seq_length, batch_size, feature_dim)，即第一维是序列长度，第二维是批量大小，第三维是特征维度
# batch_first=True表示输入和输出数据的形状为 (batch_size, seq_length, feature_dim)，即第一维是批量大小，第二维是序列长度，第三维是特征维度。
# 在构建深度学习模型时，选择 batch_first 可以有效地提高数据处理的效率和准确性。
cell_2 = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True) # 定义RNN模型，batch_first=True表示输入数据的第一个维度是batch_size

# 生成随机数据inputs,因为前面batch_first=True，所以这里将batch_size和sequence_length调换位置
dataset_BatchSize_True = torch.randn(batch_size, sequence_length, input_size) # 随机生成一个序列数据，shape=(3, 1, 4)
dataset_BatchSize_False = torch.randn(sequence_length, batch_size, input_size) # 随机生成一个序列数据，shape=(3, 1, 4)
# 初始化隐藏层的状态为全部为0的张量h0
#hidden_1 = torch.zeros(batch_size, hidden_size) # 隐藏层的初始状态，shape=(1, 2)
hidden_2 = torch.zeros(num_layer, batch_size, hidden_size) # 隐藏层的初始状态，shape=(2, 1, 2)

# # 循环遍历每个时间步的输入数据
# for idx, input in enumerate(dataset):
#     print('=' * 20, idx, '=' * 20)
#     print('input:', input.shape) # input: torch.Size([1, 4])

#     hidden = cell(input, hidden) # 计算隐藏层的状态

#     print('output:', hidden.shape) # hidden: torch.Size([1, 2])
#     print(hidden)

# 计算隐藏层的状态hN和输出(张量)
# out, hidden = cell_1(dataset_BatchSize_False, hidden_1)
out, hidden_output = cell_2(dataset_BatchSize_True, hidden_2) # 计算隐藏层的状态和输出

print('out:', out.shape) 
print(out) # out是最后一个时间步的输出
print('hidden:', hidden_output.shape)
print(hidden_output) # hidden是最后一个时间步的隐藏层状态