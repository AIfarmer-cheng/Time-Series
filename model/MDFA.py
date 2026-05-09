import torch
from torch import nn
import torch.nn.functional as F

# 通道注意力：特征维度
class tongdao(nn.Module):
    def __init__(self, in_channel): # 输入通道数
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出1*1的特征图，一个平均值。得到每个通道的全局特征，[b,c,1,1]
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于降维（通道数变1），[b,1,1,1]
        self.relu = nn.ReLU(inplace=True) # <=0输出0，>0直接输出

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x) # 相當於每個通道一個權重
        y = self.fc(y)
        y = self.relu(y)
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # 让y的H和W等于x的，[b,1,h,w]
        return x * y.expand_as(x)  # 让y的通道数等于x的，[b,c,h,w] * [b,c,h,w]

# 空间注意力：高度宽度
class kongjian(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于产生空间激励，[b,1,h,w]
        self.sigmoid = nn.Sigmoid()  # 映射到(0,1)

    def forward(self, x):
        y = self.Conv1x1(x) # 相當於每個空間位置(h,w)一個權重
        y = self.sigmoid(y)
        return x * y  # 将空间权重应用到输入x上，实现空间激励，[b,c,h,w] * [b,1,h,w]

# 把空间和通道分别提取的特征合并起来
class hebing(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)
        self.kongjian = kongjian(in_channel)

    def forward(self, U):
        U_kongjian = self.kongjian(U)
        U_tongdao = self.tongdao(U)
        return torch.max(U_tongdao, U_kongjian)  # 取两者的逐元素最大值，结合通道和空间激励

# 多尺度空洞融合注意力
class MDFA(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):# 输入和输出的通道数，rate空洞率，bn_mom批归一化的动量
        super(MDFA, self).__init__()
        # 第一分支：使用1x1卷积，保持通道维度不变，不使用空洞
        # stride步長1，padding表示在周圍補0
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 卷积核（参数量）永远是3*3个，感受野扩大为2*dilation+1
        self.branch2 = nn.Sequential( # 第二分支：使用3x3卷积，步长1，空洞率为6，可以增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True), # 根据公式padding=dilation为了让输入输出尺寸不变
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential( # 第三分支：使用3x3卷积，空洞率为12，进一步增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(# 第四分支：使用3x3卷积，空洞率为18，最大化感受野的扩展
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True) # 第五分支：全局特征提取，使用全局平均池化后的1x1卷积处理
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential( # 合并所有分支的输出，并通过1x1卷积降维
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing=hebing(in_channel=dim_out*5)# 整合通道和空间特征的合并模块

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 4个分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 第5个分支，全局特征提取
        global_feature = torch.mean(x, 2, True) # 在第2个维度上求均值，并保持维度数量不变
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        # 變回row * col大小
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # cat拼接，在通道的維度上32*5
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 应用合并模块进行通道和空间特征增强
        mdfa=self.Hebing(feature_cat)
        # 逐元素相乘加權
        mdfa_feature_cat=mdfa*feature_cat
        # 最终输出经过降维处理
        result = self.conv_cat(mdfa_feature_cat)

        return result


if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)  # 随机生成输入数据
    model = MDFA(dim_in=32,dim_out=32)  # 实例化模块
    output = model(input)  # 将输入通过模块处理
    print(output.shape)  # 输出处理后的数据形状