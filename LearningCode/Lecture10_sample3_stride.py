# stride步长

import torch
# 1个batch，1个通道，5x5的图像
input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]
input = torch.Tensor(input).view(1, 1, 5, 5) # B * C * H * H

# 卷积实际上仍然是一个线性模型，只不过是对输入数据进行了一些变换
conv_layer = torch.nn.Conv2d(in_channels=1,# 因为输入数据是单通道的，所以输入通道数为1
                             out_channels=1, # 输出通道数为1
                             kernel_size=3, 
                             stride=2, # 步长为2
                             bias=False) # 不使用偏置项
# 卷积核的输出通道数和输入通道数都为1，大小为3x3,
kernel = torch.Tensor([1,2,3,
                       4,5,6,
                       7,8,9]).view(1, 1, 3, 3) 

# 将卷积核的权重赋值到卷积层中
conv_layer.weight.data = kernel.data

output = conv_layer(input) # 进行卷积操作

# # 打印输入的形状:torch.Size([1, 1, 5, 5])
print("Input shape:", input.shape) 
# 打印输出的形状:torch.Size([1, 1, 2, 2])
# 输出的形状是(1, 1, 2, 2)，因为步长为2，所以输出的宽度和高度分别为输入的宽度和高度除以2
print("Output shape:", output.shape)