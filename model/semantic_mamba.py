"""
语义增强Mamba — 精简版
=====================
RadarMamba提取雷达特征 → SemanticEncoder编码语言先验 → Cross-Attention融合 → 分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mamba_radar import RadarMambaEncoder
from model.semantic_encoder import SemanticEncoder


class SemanticMamba(nn.Module):
    def __init__(self, num_classes: int = 100, d_model: int = 256,
                 num_mamba_blocks: int = 8, use_semantic: bool = True):
        super().__init__()
        self.use_semantic = use_semantic

        # 雷达编码器
        self.radar_encoder = RadarMambaEncoder(
            feature_dim=d_model, d_model=d_model, num_blocks=num_mamba_blocks)

        # 语义编码器
        self.semantic_encoder = SemanticEncoder(d_model=d_model)

        # 跨模态Cross-Attention: Q=雷达, K/V=语义
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True)

        # LayerNorm
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in [self.classifier]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """x: (B,2,80,128) 雷达数据, labels: (B,) 类别标签 → logits: (B,num_classes)"""
        # 1. 雷达特征: (B, d_model)
        radar_feat = self.radar_encoder(x)

        if self.use_semantic and labels is not None:
            # 2. 语义特征: (B, 6, d_model)
            sem_tokens = self.semantic_encoder(labels)

            # 3. Cross-Attention: Q=雷达, K/V=语义
            q = self.norm_q(radar_feat.unsqueeze(1))        # (B, 1, d_model)
            kv = self.norm_kv(sem_tokens)                    # (B, 6, d_model)
            fused, _ = self.cross_attn(q, kv, kv)            # (B, 1, d_model)
            fused = self.norm_out(radar_feat.unsqueeze(1) + fused)
            fused = fused.squeeze(1)                          # (B, d_model)
        else:
            fused = radar_feat

        # 4. 分类
        logits = self.classifier(fused)
        return logits
