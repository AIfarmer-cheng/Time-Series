import torch
import torch.nn as nn
import torch.nn.functional as F

# 随机使残差分支放大or完全丢弃
# 提高鲁棒性，正则化（防过拟合），缓解梯度。类比团队中随机让某些成员休息，迫使其他成员承担更多责任，提升团队整体韧性
def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training: # 残差分支失效的概率
        return x
    keep_prob = 1 - drop_prob # 保留概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 一个batch大小的元组+除去batch维度全1的元组[batch_size, 1, 1, 1]
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # 随机生成shape形状0-1之间的张量
    random_tensor.floor_()  # 二值化向下取整，shape形状张量的值只有01
    output = x.div(keep_prob) * random_tensor # x/保留概率进行数值放大
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True) # 可学习可训练缩放参数[channels]
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True) # 可学习可训练偏置参数[channels]
        self.eps = eps # 数据稳定性参数，防止除0
        self.data_format = data_format # 支持两种数据格式
        if self.data_format not in ["channels_last", "channels_first"]: # 格式验证，通道在前[batch_size, channels, height, width]
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # [batch_size, channels, height, width]
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True) # 均值
            var = (x - mean).pow(2).mean(1, keepdim=True) # 方差，pow平方
            x = (x - mean) / torch.sqrt(var + self.eps) # 归一化，标准差=sqrt(方差)
            # [channels,1,1]，对每个批次每个通道里的每个元素*weight+bias
            x = self.weight[:, None, None] * x + self.bias[:, None, None] # 缩放和平移
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6): # dim输入通道数
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise深度可分离卷积，大感受野
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 等价1*1卷积，倒残差，先扩再缩
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # 层缩放
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    # 深度卷积
    # layernorm
    # 线性层
    # 激活
    # 线性层
    # 层缩放
    # 残差连接+随机深度
    def forward(self, x: torch.Tensor):
        shortcut = x # 残差
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)   # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x      # x中每个元素乘以gamma
        x = x.permute(0, 3, 1, 2)   # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):

    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        # depths每个stage的Block数量
        # dims每个stage的通道数
        super().__init__()
        # 下采样层
        self.downsample_layers = nn.ModuleList()
        # 1个主干
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), # 下采样4倍[B,3,224,224]->[B,96,56,56]
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        # 3个中间下采样层
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)) # 每次空间尺寸减半，通道数加倍
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # depths[3,3,9,3],dp_rates递增的18个值
        cur = 0
        # 4个stage
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        # 最后一层
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        # 权重初始化
        self.apply(self._init_weights) # apply遍历所有层
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)): # 只对卷积层和线性层初始化
            nn.init.trunc_normal_(m.weight, std=0.2) # std标准差
            nn.init.constant_(m.bias, 0) # 权重为正太分布，大部分在[-0.4,0.4]之间

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # global average pooling对最后两个维度求平均,(N,C,H,W)->(N,C)

    # stem
    # stage1(3个Block)
    # 下采样1
    # stage2(3个Block)
    # 下采样2
    # stage3(9个Block)
    # 下采样3
    # stage4(3个Block)
    # 归一化(全局平均池化)
    # 分类
    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnext_tiny(num_classes: int = 1000, in_chans: int = 3):

    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     in_chans=in_chans)
    return model


def convnext_small(num_classes: int = 1000, in_chans: int = 3):

    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     in_chans=in_chans)
    return model


def convnext_base(num_classes: int = 1000, in_chans: int = 3):

    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes,
                     in_chans=in_chans)
    return model


def convnext_large(num_classes: int = 1000, in_chans: int = 3):

    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes,
                     in_chans=in_chans)
    return model


def convnext_xlarge(num_classes: int = 1000, in_chans: int = 3):

    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes,
                     in_chans=in_chans)
    return model
