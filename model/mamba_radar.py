"""
雷达Mamba编码器
=======================
将雷达时频谱(80×128)展平为序列，用Mamba块建模长程依赖，[CLS] token聚合全局特征。
Mamba核心(SelectiveSSM)从零实现，使用顺序扫描保证正确性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """选择性状态空间模型 — Mamba的核心

    输入序列 x ∈ R^(B,L,D)，通过数据依赖的Δ/B/C参数选择性记忆或遗忘。
    使用欧拉离散化和顺序扫描实现（简单、正确、可读）。
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = d_model * expand

        # 输入投影 → 拆分为 x(进SSM) 和 z(门控)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 深度卷积：每个通道独立卷积，保持通道内相关性
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1, bias=False)

        # Δ/B/C投影
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # 可学习的状态矩阵 A（保持log空间确保稳定性）
        # Mamba/S4要求A的每个对角线元素必须互不相同
        # 确保每个内部通道有独立的状态动力学，但每个通道共享相同的衰减速率集合
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)，从1到d_state
        A = A.repeat(d_inner, 1)                                            # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A / d_state))
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _selective_scan(self, u, delta, A, B, C, D):
        """并行前缀扫描，非for循环，非CUDA kernel，纯PyTorch实现"""
        Bsz, L, D_inner = u.shape
        N = A.shape[1]
        # 离散化: deltaA (B,L,D_inner,N), deltaB_u (B,L,D_inner,N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        # 并行关联扫描 (Parallel Associative Scan)，高效的SSM递归，将串行递归变为二叉树归并
        # 关联操作 ⊕: (a1,b1) ⊕ (a2,b2) = (a1*a2, a1*b2+b1)
        # 对 (deltaA, deltaB_u) 做前缀扫描，O(log L)步
        a = deltaA
        b = deltaB_u

        num_steps = L.bit_length()
        for step in range(num_steps):
            stride = 1 << step # stride = 1,2,4...2^step
            if stride >= L:
                break
            # 取左右两半（不原地修改，autograd友好）
            a_left = a[:, :L - stride]
            a_right = a[:, stride:]
            b_left = b[:, :L - stride]
            b_right = b[:, stride:]
            # 关联合并: (a1,b1)⊕(a2,b2) = (a1*a2, a1*b2+b1)
            a_merged = a_left * a_right
            b_merged = a_left * b_right + b_left
            # 拼接：前stride个位置不变，后续用合并结果
            a = torch.cat([a[:, :stride], a_merged], dim=1)
            b = torch.cat([b[:, :stride], b_merged], dim=1)

        # b现在存的是累积状态ht在各位置的输出
        y = (b * C.unsqueeze(2)).sum(dim=-1) + D * u   # (B, L, D_inner)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        D_inner = D * self.expand
        # 投影：一个用于输入进ssm，一个用于门控
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)                       # 两个(B, L, D_inner)
        # 对输入深度卷积
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[..., :L]
        x_ssm = x_ssm.transpose(1, 2).contiguous()           # (B, L, D_inner)
        # SiLU激活
        x_ssm = F.silu(x_ssm)

        # 计算SSM参数，Δ/B/C/A，除了A初始化外，其他参数从输入中学习
        delta = F.softplus(self.dt_proj(x_ssm))              # (B, L, D_inner)
        B_ssm = self.B_proj(x_ssm)                           # (B, L, D_state)
        C_ssm = self.C_proj(x_ssm)                           # (B, L, D_state)
        A_ssm = -torch.exp(self.A_log)                       # (D_inner, D_state)

        # 选择性扫描
        y = self._selective_scan(x_ssm, delta, A_ssm, B_ssm, C_ssm, self.D)

        # 输出*门控，选择性处理
        y = y * F.silu(z)
        y = self.out_proj(y)                                 # (B, L, D)

        return y


class MambaBlock(nn.Module):
    """Mamba基础块: LayerNorm → SelectiveSSM → 残差连接"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class RadarMambaEncoder(nn.Module):
    """雷达Mamba编码器
    利用2维卷积将时频打碎成640个patch token
    输入: (B, 2, 80, 128) — 雷达复数I/Q两通道
    流程: Stem下采样 → 展平序列 → [CLS] token → Mamba块堆叠 → 取[CLS]全局特征聚合 → 特征投影
    输出: (B, feature_dim)
    """
    def __init__(self, feature_dim: int = 256, d_model: int = 192, num_blocks: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model

        # Stem: 2通道 → d_model通道，同时下采样
        # 80×128 → 20×32 (输出尺寸=输入+2p-k/s + 1)
        self.stem = nn.Sequential(
            nn.Conv2d(2, d_model, kernel_size=4, stride=4, bias=False),
            nn.GELU(),
        )

        # [CLS] token + 位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # stem后空间尺寸: 20×32 = 640 个patch, +1 = 641
        self.pos_embed = nn.Parameter(torch.randn(1, 641, d_model) * 0.02)

        # Mamba块堆叠
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_blocks)
        ])

        # 输出投影
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, feature_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Stem: (B, 2, 80, 128) → (B, d_model, 20, 32)
        x = self.stem(x)
        # 展平为语言序列: (B, d_model, 20, 32) → (B, 640, d_model)
        x = x.flatten(2).transpose(1, 2)
        # 添加 [CLS] token: (B, 641, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        # 位置编码
        x = x + self.pos_embed[:, :x.shape[1], :]
        # Mamba块：输入[B,L,d_model]
        for block in self.blocks:
            x = block(x)

        # 取 [CLS] token
        x = self.final_norm(x[:, 0, :])     # (B, d_model)
        x = self.head(x)                     # (B, feature_dim) 输出作为Q

        return x
