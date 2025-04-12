import torch
# 定义输入和输出的通道数
in_channels, out_channels = 5, 10
# 定义输入的宽度和高度
width, height = 100, 100
# 定义卷积核的大小为3x3（一般情况是正方形）
kernel_size = 3
batch_size = 1

# 随机生成一个输入张量，实际应用时不需要随机生成
input = torch.randn(batch_size, 
                    in_channels, 
                    height,
                    width) 
# 卷积层
# 不在乎输入数据的宽度和高度，因为是复用了输入数据的宽度和高度
conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels, 
                             kernel_size=kernel_size) 

output = conv_layer(input) # 进行卷积操作

# 打印输出的形状:torch.Size([1, 5, 100, 100])
print("Input shape:", input.shape) 
# 打印输出的形状:torch.Size([1, 10, 98, 98])
# 因为卷积核的大小为3x3，所以输出的宽度和高度会减少2
print("Output shape:", output.shape) 
# 打印卷积核的形状:out_channels, in_channels, kernel_size, kernel_size
# 卷积核的形状是(10, 5, 3, 3)，表示有10个输出通道，每个输出通道有5个输入通道，每个卷积核的大小为3x3
print("Kernel shape:", conv_layer.weight.shape) 


