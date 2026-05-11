import torch
import torch.nn as nn
import torch.nn.functional as F


# 使用cuda并行训练，串行推理
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        # 输入投影生成 selective 参数
        self.x_proj = nn.Linear(d_model, 3 * d_state)
        # 状态矩阵 A（可学习）
        self.A = nn.Parameter(torch.randn(d_state))
        # 输出映射
        self.out_proj = nn.Linear(d_state, d_model)

    def forward(self, x):
        B, L, D = x.shape
        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []
        params = self.x_proj(x)  # (B, L, 3*d_state)
        delta, B_t, C_t = torch.chunk(params, 3, dim=-1)
        delta = F.softplus(delta) # 取非负值

        # S4模型
        # 遍历每个时间步进行采样，离散化可选择性SSM
        # 严格并行扫描，这里串行
        for t in range(L):
            dt = delta[:, t]
            bt = B_t[:, t]
            ct = C_t[:, t]
            xt = x[:, t]

            # A_bar之前所有时刻的信息浓缩，严格使用HiPPO初始化
            A_bar = torch.exp(-dt * self.A)
            h = A_bar * h + bt * xt.mean(dim=-1, keepdim=True) # ht是系统行为 ht-1是潜在隐藏状态，xt当前输入
            y = ct * h

            outputs.append(y) # [[B,d_model]...L个]
        # 拼接回[B,L,d_model]
        y = torch.stack(outputs, dim=1)

        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        # 深度卷积，每个通道分别对应一个通道数为1大小为4的卷积核，1D卷积就是在序列上滑动的窗口
        # 区别普通卷积：一个卷积核的通道数等于输入通道数，输出通道数等于卷积核个数
        self.conv1d = nn.Conv1d( # 使用A_bar和B_bar初始化kernel
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model # 深度标志
        )
        self.ssm = SelectiveSSM(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1) # 门控机制，因为AB训练完就定死了，所以需要门控来选择性处理
        x_conv = self.conv1d(x.transpose(1, 2))
        x_conv = x_conv[:, :, :x.size(1)].transpose(1, 2)
        # SSM
        y = self.ssm(x_conv)
        # gating
        y = y * torch.sigmoid(gate)

        return self.out_proj(y) + residual


class Mamba(nn.Module):
    def __init__(self, vocab_size, d_model=128, d_state=16, n_layers=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model) # [2,64]->[2,64,128]
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

model = Mamba(
    vocab_size=5000,
    d_model=128,
    d_state=16,
    n_layers=4
)
x = torch.randint(0, 5000, (2, 64)) # 词汇表5000个词，初始输入[2,64]
logits = model(x)
print(logits.shape)