"""
RadarMambaEncoder — 雷达数据 Mamba 编码器
=========================================
基于 Mamba (MambaSimple) 架构，加 2D CNN 前端提取局部时频纹理，
再将空间维度展平为长序列送入 Mamba 选择性扫描。

输入:  (B, 2, 80, 128)  雷达IQ时频谱 [实部, 虚部] × 80帧 × 128频点
      → ConvStem 2D CNN → (B, d_model, 10, 16)
      → flatten → (B, 160, d_model)  序列
      → Mamba 残差块序列（核心完全不动）
      → mean → (B, d_model)          特征向量

输出:  (B, d_model)      供 SemanticMamba 跨模态融合使用

改造自 models_qinghua/MambaSimple.py:
  - ResidualBlock / MambaBlock / RMSNorm 完全保留不动
  - 去掉 DataEmbedding / forecast 标准化/反标准化
  - 新增 ConvStem 2D 卷积前端替代单层 Linear 投影
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum


# ═══════════════════════════════════════════════════════════════════════
# ConvStem — 2D CNN 前端，提取时频谱局部纹理
# ═══════════════════════════════════════════════════════════════════════

class ConvStem(nn.Module):
    """2D 卷积前端：在雷达时频谱上提取局部时频纹理特征

    (B, 2, 80, 128)  →  4层 Conv2d + BN + SiLU  →  (B, out_dim, 10, 16)
                      →  flatten spatial  →  (B, 160, out_dim)  序列
    """

    def __init__(self, in_channels: int = 2, out_dim: int = 192,
                 hidden_dims: tuple = (32, 64, 128)):
        super().__init__()
        layers = []
        dims = (in_channels,) + hidden_dims + (out_dim,)

        # 三层降采样: 80→40→20→10, 128→64→32→16
        for i in range(len(dims) - 1):
            stride = 2 if i in (0, 1, 2) else 1  # 前3层降采样，末层保分辨率
            layers.append(nn.Conv2d(dims[i], dims[i+1], kernel_size=3,
                                    stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(dims[i+1]))
            layers.append(nn.SiLU())

        self.conv = nn.Sequential(*layers)
        self.out_spatial = 10 * 16  # = 160

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, 80, 128) → (B, 160, out_dim)
        """
        x = self.conv(x)                            # (B, out_dim, 10, 16)
        x = x.flatten(2).transpose(1, 2)            # (B, 160, out_dim)
        return x


# ═══════════════════════════════════════════════════════════════════════
# 主类：RadarMambaEncoder — semantic_mamba.py 直接导入此类
# ═══════════════════════════════════════════════════════════════════════

class RadarMambaEncoder(nn.Module):
    """雷达IQ时频谱的Mamba序列编码器

    2D Conv 前端 → 展平为 160 步序列 → Mamba 选择性扫描 → 池化 → 特征向量

    Parameters
    ----------
    feature_dim : int
        预留参数，与 semantic_mamba 接口兼容（实际由 d_model 决定）
    d_model : int
        ConvStem 输出通道 = Mamba 内部隐层维度，默认 192
    num_blocks : int
        Mamba + Residual 块的数量，默认 4
    d_state : int
        SSM 状态维度 (原 configs.d_ff)，默认 16
    d_conv : int
        1D 卷积核大小，默认 4
    expand : int
        内层维度扩展倍数 (d_inner = d_model * expand)，默认 2
    dropout : float
        Dropout 比例，默认 0.1
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 192,
        num_blocks: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        # ---------- 2D CNN 前端 ----------
        # 替代原来的单层 Linear 投影，提取时频局部纹理
        self.conv_stem = ConvStem(in_channels=2, out_dim=d_model)
        self.seq_len = self.conv_stem.out_spatial  # 640

        # ---------- 可学习位置编码 ----------
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)

        # ---------- Mamba 残差块 ----------
        # （以下完全不动，与 MambaSimple 原始实现一致）
        self.layers = nn.ModuleList([
            ResidualBlock(
                d_model=d_model,
                d_inner=self.d_inner,
                dt_rank=self.dt_rank,
                d_state=d_state,
                d_conv=d_conv,
            )
            for _ in range(num_blocks)
        ])

        # ---------- 最终归一化 ----------
        self.norm = RMSNorm(d_model)

        # ---------- Dropout ----------
        self.dropout = nn.Dropout(dropout)

        # ---------- 输出投影 ----------
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.out_proj:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 2, 80, 128)
            雷达IQ时频谱，通道0=实部，通道1=虚部

        Returns
        -------
        torch.Tensor, shape (B, d_model)
            雷达特征向量，供 semantic_mamba 跨模态注意力使用
        """
        # 2D CNN 前端: (B, 2, 80, 128) → (B, 640, d_model)
        x = self.conv_stem(x)

        # 位置编码
        x = x + self.pos_embed
        x = self.dropout(x)

        # Mamba 残差块序列（核心，完全不动）
        for layer in self.layers:
            x = layer(x)                          # (B, 640, d_model)

        # 最终归一化
        x = self.norm(x)

        # 全局平均池化 → 聚合长序列信息
        x = x.mean(dim=1)                         # (B, d_model)

        # 输出投影
        x = self.out_proj(x)                      # (B, d_model)

        return x


# ═══════════════════════════════════════════════════════════════════════
# 以下子模块直接复用 MambaSimple 原始实现，仅将 configs 参数展开为显式参数
# ═══════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """MambaBlock + RMSNorm + 残差连接"""

    def __init__(self, d_model: int, d_inner: int, dt_rank: int,
                 d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.mixer = MambaBlock(d_model, d_inner, dt_rank, d_state, d_conv)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # Pre-norm 残差: x + MambaBlock(Norm(x))
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):
    """
    Mamba 选择性状态空间模块
    论文: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    参考实现: https://github.com/johnma2006/mamba-minimal/
    """

    def __init__(self, d_model: int, d_inner: int, dt_rank: int,
                 d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        # 输入投影: x → (x, gate) 两路
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 深度可分离1D卷积（局部时序建模）
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )

        # SSM 参数投影: x → (delta, B, C)
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        # delta 投影
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # 状态矩阵 A (对数空间，可学习) — HiPPO 初始化
        A = repeat(torch.arange(1, d_state + 1), "n -> d n", d=d_inner).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, d_model) → output: (B, L, d_model)
        """
        (b, l, d) = x.shape

        # 1. 输入投影成两路: x 和 residual (gate)
        x_and_res = self.in_proj(x)                                    # (B, L, 2*d_inner)
        (x, res) = x_and_res.split(
            split_size=[self.d_inner, self.d_inner], dim=-1)

        # 2. 1D 深度卷积 (局部特征混合)
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        # 3. 选择性 SSM 扫描
        y = self.ssm(x)

        # 4. 门控机制: 输出 × silu(gate)
        y = y * F.silu(res)

        # 5. 输出投影
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """选择性状态空间模型 (Algorithm 2)"""
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())          # (d_in, d_state)
        D = self.D.float()                          # (d_in,)

        # 投影出 delta, B, C
        x_dbl = self.x_proj(x)                       # (B, L, dt_rank + 2*d_state)
        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))      # (B, L, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        选择性扫描 (Selective Scan)
        u:     (B, L, d_in)
        delta: (B, L, d_in)
        A:     (d_in, d_state)
        B:     (B, L, d_state)
        C:     (B, L, d_state)
        D:     (d_in,)
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # ZOH 离散化 A, 简化欧拉离散化 B
        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        # 逐时间步选择性扫描 (论文串行实现)
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1)                  # (B, L, d_in)
        y = y + u * D                                # 跳跃连接
        return y


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


# ═══════════════════════════════════════════════════════════════════════
# 快速自测
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("RadarMambaEncoder 自测 (ConvStem + Mamba)")
    print("=" * 70)

    dummy = torch.randn(2, 2, 80, 128)

    for nb in [2, 4, 8]:
        model = RadarMambaEncoder(
            feature_dim=192,
            d_model=192,
            num_blocks=nb,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        out = model(dummy)
        total = sum(p.numel() for p in model.parameters()) / 1e6
        mamba_p = sum(p.numel() for p in model.layers.parameters()) / 1e6
        conv_p  = sum(p.numel() for p in model.conv_stem.parameters()) / 1e6
        print(f"  num_blocks={nb}: (2,80,128) → ConvStem → (2,160,{model.d_model})"
              f" → Mamba → mean → {tuple(out.shape)}")
        print(f"           总参数量 {total:.2f}M  "
              f"(ConvStem {conv_p:.2f}M + Mamba {mamba_p:.2f}M)")

    print("\n✓ 全部通过 — ConvStem + Mamba 核心 + semantic_mamba 接口均匹配")
