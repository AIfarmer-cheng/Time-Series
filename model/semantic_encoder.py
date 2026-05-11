"""
汉字语义编码器 — 精简版
=======================
从预生成的JSON加载100个汉字的语言学特征，将6个维度各自嵌入为token，
输出(B, 6, d_model)供Cross-Attention使用。

六大维度: 拼音、声母、韵母、声调、部首、笔画数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import csv
from pathlib import Path


class SemanticEncoder(nn.Module):
    def __init__(self, d_model: int = 256, features_json: str = None):
        super().__init__()

        # ---- 加载特征数据库 ----
        if features_json is None:
            features_json = Path(__file__).resolve().parent.parent / \
                           'chinese_radar_dataset' / 'character_features.json'

        with open(features_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = data['vocab_info']['vocab_sizes'] # 存储每个维度的特征数量
        self.features = data['features'] # 所有汉字的特征信息，包括拼音、声母、韵母、声调、部首、笔画数

        # 建立 char_index → 特征索引 映射
        chars_csv = Path(__file__).resolve().parent.parent / \
                   'chinese_radar_dataset' / 'characters.csv'
        with open(chars_csv, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f)) # 每一行变成字典

        self._feat_idx = {}
        for row in rows:
            ch = row['char']
            cid = int(row['char_index'])
            feat = next((f for f in self.features if f['char'] == ch), None) # 用汉字去json中找对应它的特征信息
            if feat:
                self._feat_idx[cid] = {
                    'pinyin': feat['pinyin_idx'], # 同音不同形的字也应该有相同的拼音索引
                    'initial': feat['initial_idx'], # 提取5个信息对应在词汇表中的索引
                    'final': feat['final_idx'],
                    'tone': feat['tone_idx'],
                    'radical': feat['radical_idx'],
                    'stroke': feat['stroke_idx'],
                }

        # ---- 6个Embedding层 ----
        self.pinyin_embed  = nn.Embedding(self.vocab['pinyin'], d_model)
        self.initial_embed = nn.Embedding(self.vocab['initial'], d_model)
        self.final_embed   = nn.Embedding(self.vocab['final'], d_model)
        self.tone_embed    = nn.Embedding(self.vocab['tone'], d_model)
        self.radical_embed = nn.Embedding(self.vocab['radical'], d_model)
        self.stroke_embed  = nn.Embedding(self.vocab['stroke'], d_model)

        # 初始化
        for emb in [self.pinyin_embed, self.initial_embed, self.final_embed,
                     self.tone_embed, self.radical_embed, self.stroke_embed]:
            nn.init.trunc_normal_(emb.weight, std=0.02)

    def forward(self, char_indices: torch.Tensor) -> torch.Tensor:
        """char_indices: (B,) 类别标签0-99 → 语义tokens: (B, 6, d_model)"""
        B = char_indices.shape[0] # 0-99的序号
        device = char_indices.device

        # 批量收集各维度索引
        batch_pinyin = torch.zeros(B, dtype=torch.long, device=device)
        batch_init = torch.zeros(B, dtype=torch.long, device=device)
        batch_final = torch.zeros(B, dtype=torch.long, device=device)
        batch_tone = torch.zeros(B, dtype=torch.long, device=device)
        batch_rad = torch.zeros(B, dtype=torch.long, device=device)
        batch_strk = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(B):
            cid = char_indices[i].item()
            feats = self._feat_idx.get(cid, self._feat_idx.get(0))
            batch_pinyin[i] = feats['pinyin']
            batch_init[i] = feats['initial'] # 找到第i个汉字的声母索引
            batch_final[i] = feats['final']
            batch_tone[i] = feats['tone']
            batch_rad[i] = feats['radical']
            batch_strk[i] = feats['stroke']

        # 6路嵌入，将一个索引值embedding成d_model的向量
        tok_pinyin  = self.pinyin_embed(batch_pinyin)          # (B, d_model)
        tok_initial = self.initial_embed(batch_init)            # (B, d_model)
        tok_final   = self.final_embed(batch_final)             # (B, d_model)
        tok_tone    = self.tone_embed(batch_tone)               # (B, d_model)
        tok_radical = self.radical_embed(batch_rad)             # (B, d_model)
        tok_stroke  = self.stroke_embed(batch_strk)             # (B, d_model)

        # 堆叠为6个语义token
        tokens = torch.stack([
            tok_pinyin, tok_initial, tok_final,
            tok_tone, tok_radical, tok_stroke
        ], dim=1)                                               # (B, 6, d_model)

        return tokens
