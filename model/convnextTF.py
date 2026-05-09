import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MDFA import MDFA
from model.GGCA import GGCA

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


# 侧向抑制连接（Lateral Inhibition Connections）
# 思想：借鉴神经科学中的侧向抑制机制，通过局部竞争机制增强特征的稀疏性和判别性
# 每个神经元不仅接收来自前一层的输入，还接收来自相邻神经元的抑制信号
# 实现：output = input - α * (local_avg(input))
class LateralInhibition(nn.Module):
    """
    实现方式：
    - 计算输入特征的局部平均值
    - 使用可学习参数α控制抑制强度
    """
    def __init__(self, dim, kernel_size=3, alpha_init=0.1): # 抑制强度α的初始值
        super(LateralInhibition, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        # logit(p)=log(p/1-p)，再ones爲shape為[1]的張量，最後Parameter變爲一個可訓練參數
        self.alpha_logit = nn.Parameter(torch.ones(1) * self._logit(alpha_init), requires_grad=True)
        # 局部平均池化（因为3*3卷积核，stride=1）
        self.local_avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=1)
    
    def _logit(self, p):
        """将概率p转换为logit值"""
        p_tensor = torch.tensor(float(p))
        return torch.log(p_tensor / (1 - p_tensor + 1e-8))
    
    def forward(self, x: torch.Tensor):
        local_avg = self.local_avg(x)
        # 获取抑制强度α（通过Sigmoid限制在0-1之间）
        alpha = torch.sigmoid(self.alpha_logit)
        # 侧向抑制：output = input - α * local_avg(input)
        # 突出重要特征，增强对比，提高判别性，抑制噪声
        output = x - alpha * local_avg
        return output


# 自组织深度：可学习的跳跃门
# 思想：让网络学会应该跳过哪些层，而不是随机丢弃
# 使用可学习参数α，通过Sigmoid限制在0-1之间，控制主路径和恒等路径的混合比例
# α接近1表示"使用该层"，α接近0表示"跳过该层"
class GatePath(nn.Module):
    def __init__(self):
        super(GatePath, self).__init__()
        self.gate_alpha = nn.Parameter(torch.ones(1), requires_grad=True) # 可學習α初始化为1
    
    def forward(self, x):
        gate_value = torch.sigmoid(self.gate_alpha) # 通過sigmoid固定其為0-1
        return x * gate_value # 幾乎保留或幾乎全打開
    
    def get_gate_value(self):
        """获取当前门的激活值，用于计算正则项"""
        return torch.sigmoid(self.gate_alpha) # 收集‘門值’，訓練時會加到損失loss裏面去，所以鼓勵關門


# 類似Transformer
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
        # [N, H, W, C]對C通道歸一化
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # [batch_size, channels, height, width]
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True) # 均值μ
            var = (x - mean).pow(2).mean(1, keepdim=True) # 方差σ^2，pow平方
            x = (x - mean) / torch.sqrt(var + self.eps) # 標準化，x-μ/sqrt(σ^2+eps)
            # [channels,1,1]，对每个批次每个通道里的每个元素*weight+bias
            x = self.weight[:, None, None] * x + self.bias[:, None, None] # 可學習缩放和偏置，ax+b
            return x


# drop_rat=0不啓用隨機深度
class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6, use_gate=True, 
                 use_lateral_inhibition=True, inhibition_kernel_size=3): # dim输入通道数
        super().__init__()
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise深度卷积，大感受野（已替换为MDFA模块）
        self.mdfa = MDFA(dim_in=dim, dim_out=dim, rate=1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 等价1*1卷积形成深度可分离卷积，倒残差，维度先扩再缩
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # 层缩放，可學習
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        # 自组织深度
        self.use_gate = use_gate
        self.gate_path = GatePath() if use_gate else nn.Identity()
        # 侧向抑制连接
        self.use_lateral_inhibition = use_lateral_inhibition
        self.lateral_inhibition = LateralInhibition(dim=dim, kernel_size=inhibition_kernel_size) if use_lateral_inhibition else nn.Identity()

    # MDFA模块
    # layernorm
    # 线性层
    # 激活
    # 线性层
    # 层缩放
    # 侧向抑制
    # 残差连接+自组织深度
    def forward(self, x: torch.Tensor):
        shortcut = x # 原始数据
        # x = self.dwconv(x)
        x = self.mdfa(x)            # 使用MDFA模块进行多尺度空洞融合注意力
        x = x.permute(0, 2, 3, 1)   # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)            
        x = self.pwconv1(x)         # 对维度开拓展到高维，再压缩回原维
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:  # 不想一上來就把shortcut淹沒，想要慢慢學習
            x = self.gamma * x      # 讓最後一維C乘以gamma，某些通道重要某些通道不重要
        x = x.permute(0, 3, 1, 2)   # [N, H, W, C] -> [N, C, H, W]
        
        # 侧向抑制
        x = self.lateral_inhibition(x)  # [N, C, H, W]

        # x = self.drop_path(x)
        # 自组织深度
        x = self.gate_path(x)
        # 殘差鏈接
        x = shortcut + x
        return x
    
    def get_gate_value(self):
        if self.use_gate:
            return self.gate_path.get_gate_value()
        return None

# drop_path_rate=0时不启用随机深度
# use_gate=True启用自组织深度
# use_lateral_inhibition=True启用侧向抑制连接
class TFConv(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1., use_gate: bool = True, 
                 use_lateral_inhibition: bool = True, inhibition_kernel_size: int = 3):
        # depths每个stage的Block数量
        # dims每个stage的通道数
        # use_gate: 是否使用自组织深度
        # use_lateral_inhibition: 是否使用侧向抑制连接
        # inhibition_kernel_size: 侧向抑制的局部平均核大小
        super().__init__()
        self.use_gate = use_gate
        self.use_lateral_inhibition = use_lateral_inhibition
        

        # 下采样层（一共4個）
        # 就是讓通道數從2到768，讓h和w不斷除以4除以2，dim代表通道數
        self.downsample_layers = nn.ModuleList()
        # 1个主干下采样
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), # 下采样4倍[B,2,80,128]->[B,96,20,32]
                             GGCA(channel=dims[0], h=20, w=32),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        # 3个中间下采样层
        h=[10,5,2]
        w=[16,8,4]
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                                             GGCA(channel=dims[i+1], h=h[i], w=w[i])) # 下采样2倍，每次空间尺寸减半，通道数加倍
            self.downsample_layers.append(downsample_layer)


        # stages
        self.stages = nn.ModuleList()
        # 這裏都是0，不適用隨機深度
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # depths[3,3,9,3]，生成長度為sum(depths)的一維張量，數值從0到drop_path_rate
        cur = 0
        # 4个stage（每個stage包含多個Block塊）
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_rate=dp_rates[cur + j], 
                                         layer_scale_init_value=layer_scale_init_value,
                                         use_gate=use_gate,
                                         use_lateral_inhibition=use_lateral_inhibition,
                                         inhibition_kernel_size=inhibition_kernel_size) for j in range(depths[i])])
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
            if m.bias is not None:  # 只有当bias存在时才初始化（某些层可能设置了bias=False）
                nn.init.constant_(m.bias, 0) # 权重为正太分布，大部分在[-0.4,0.4]之间


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # global average pooling对最后两个维度求平均,(N,C,H,W)->(N,C)


    def get_all_gate_values(self):
        gate_values = []
        if not self.use_gate:
            return gate_values
        # 遍历所有stage中的所有Block
        for stage in self.stages:
            for block in stage:
                if isinstance(block, Block):
                    gate_val = block.get_gate_value()
                    if gate_val is not None:
                        gate_values.append(gate_val)
        
        return gate_values
    
    def compute_gate_regularization(self, lambda_reg: float = 0.01): # lambda_reg: 正则化系数，控制正则项的强度。值越大，越鼓励关闭门
        if not self.use_gate:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        gate_values = self.get_all_gate_values()
        if len(gate_values) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # 计算所有门激活值的平均值，鼓励它们接近0（关闭）
        avg_gate = torch.stack(gate_values).mean() # stack拼接成一个张量
        reg_loss = lambda_reg * avg_gate
        
        return reg_loss

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


def tfconv_tiny(num_classes: int = 1000, in_chans: int = 3, use_gate: bool = True,
                   use_lateral_inhibition: bool = True, inhibition_kernel_size: int = 3):
    model = TFConv(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     in_chans=in_chans,
                     use_gate=use_gate,
                     use_lateral_inhibition=use_lateral_inhibition,
                     inhibition_kernel_size=inhibition_kernel_size)
    return model


def tfconv_small(num_classes: int = 1000, in_chans: int = 3, use_gate: bool = True,
                    use_lateral_inhibition: bool = True, inhibition_kernel_size: int = 3):
    model = TFConv(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     in_chans=in_chans,
                     use_gate=use_gate,
                     use_lateral_inhibition=use_lateral_inhibition,
                     inhibition_kernel_size=inhibition_kernel_size)
    return model


def tfconv_base(num_classes: int = 1000, in_chans: int = 3, use_gate: bool = True,
                   use_lateral_inhibition: bool = True, inhibition_kernel_size: int = 3):
    model = TFConv(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes,
                     in_chans=in_chans,
                     use_gate=use_gate,
                     use_lateral_inhibition=use_lateral_inhibition,
                     inhibition_kernel_size=inhibition_kernel_size)
    return model


def tfconv_large(num_classes: int = 1000, in_chans: int = 3, use_gate: bool = True,
                    use_lateral_inhibition: bool = True, inhibition_kernel_size: int = 3):
    model = TFConv(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes,
                     in_chans=in_chans,
                     use_gate=use_gate,
                     use_lateral_inhibition=use_lateral_inhibition,
                     inhibition_kernel_size=inhibition_kernel_size)
    return model


def tfconv_xlarge(num_classes: int = 1000, in_chans: int = 3, use_gate: bool = True,
                     use_lateral_inhibition: bool = True, inhibition_kernel_size: int = 3):
    model = TFConv(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes,
                     in_chans=in_chans,
                     use_gate=use_gate,
                     use_lateral_inhibition=use_lateral_inhibition,
                     inhibition_kernel_size=inhibition_kernel_size)
    return model
