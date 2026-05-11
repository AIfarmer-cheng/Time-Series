"""
语义增强Mamba — 正确版（无标签泄露）
=====================
RadarMamba提取雷达特征 → SemanticEncoder提供所有类别语义原型 → 计算雷达与各类别的匹配度 → 分类

关键修正：
- 语义编码器不再接收目标标签，而是预计算所有100个类别的语义原型
- 推理时雷达特征与所有类别的语义原型计算相似度，无需知道目标标签
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
        self.num_classes = num_classes
        self.d_model = d_model

        # 雷达编码器
        self.radar_encoder = RadarMambaEncoder(
            feature_dim=d_model, d_model=d_model, num_blocks=num_mamba_blocks)

        # 语义编码器：预计算所有类别的语义原型（不依赖目标标签）
        self.semantic_encoder = SemanticEncoder(d_model=d_model, num_classes=num_classes)

        # 跨模态Cross-Attention: Q=雷达, K/V=所有类别的语义原型
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,2,80,128) 雷达数据 → logits: (B,num_classes)
        
        注意：不再接收labels参数！语义信息来自预计算的类别原型，与输入数据无关。
        """
        # 1. 雷达特征: (B, d_model)
        radar_feat = self.radar_encoder(x)

        if self.use_semantic:
            # 2. 所有类别的语义原型: (num_classes, d_model)
            # 不需要传入labels！所有类别的语义信息在初始化时已预计算好
            sem_prototypes = self.semantic_encoder()  # (num_classes, d_model)

            # 3. Cross-Attention: Q=雷达, K/V=所有类别语义原型
            # 扩展为batch维度: (1, num_classes, d_model) → (B, num_classes, d_model)
            B = radar_feat.shape[0]
            kv = self.norm_kv(sem_prototypes).unsqueeze(0).expand(B, -1, -1)  # (B, num_classes, d_model)
            q = self.norm_q(radar_feat.unsqueeze(1))  # (B, 1, d_model)

            fused, _ = self.cross_attn(q, kv, kv)  # (B, 1, d_model)
            fused = self.norm_out(radar_feat.unsqueeze(1) + fused)
            fused = fused.squeeze(1)  # (B, d_model)
        else:
            fused = radar_feat

        # 4. 分类
        logits = self.classifier(fused)
        return logits
