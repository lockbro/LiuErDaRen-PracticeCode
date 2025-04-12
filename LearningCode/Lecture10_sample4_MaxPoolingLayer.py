# 下采样subsampling
# 1. 下采样的目的是为了减少特征图的大小，从而减少计算量和内存占用。
# 2. 下采样的方式有很多种，最常用的是最大池化和平均池化。
# 最大池化MaxPooling Layer默认是2x2的池化核，步长为2，填充为0，取最大值
# 下采样是在一个通道上进行的，也就是说如果输入数据有多个通道，那么每个通道都会进行最大池化操作，得到的输出数据也是多个通道。
# 因此通道数不会发生变化，只有宽度和高度会发生变化。
import torch
# 1个batch，1个通道，5x5的图像
input = [3,4,6,5,
         2,4,6,8,
         1,6,7,8,
         9,7,4,6]
input = torch.Tensor(input).view(1, 1, 4, 4) # B * C * H * H

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 默认是2x2的池化核，步长为2，填充为0
output = maxpooling_layer(input) # 进行最大池化操作

print("Input shape:", input.shape) # 打印输入的形状:torch.Size([1, 1, 4, 4])
print("Output shape:", output.shape) # 打印输出的形状:torch.Size([1, 1, 2, 2])