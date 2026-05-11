"""
汉字语义编码器 — 无标签泄露
=======================
从预生成的JSON加载100个汉字的语言学特征，预计算所有类别的语义原型向量。
推理时不需要传入目标标签，所有类别的语义信息在初始化时已准备好。

六大维度: 拼音、声母、韵母、声调、部首、笔画数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import csv
from pathlib import Path


class SemanticEncoder(nn.Module):
    def __init__(self, d_model: int = 256, num_classes: int = 100, features_json: str = None):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

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

        # ---- 预计算所有类别的语义原型向量 ----
        # 根据char_features.json，在初始化时为每个类别生成语义向量
        # 这些向量作为可学习参数，在训练中优化
        prototypes = self._build_prototypes()  # (num_classes, d_model)
        self.class_prototypes = nn.Parameter(prototypes, requires_grad=True)

    def _build_prototypes(self):
        """根据知识库预计算所有类别的语义原型向量"""
        prototypes = []
        for cid in range(self.num_classes):
            feats = self._feat_idx.get(cid, self._feat_idx.get(0))

            # 各维度索引
            pinyin_idx  = torch.tensor(feats['pinyin'])
            initial_idx = torch.tensor(feats['initial'])
            final_idx   = torch.tensor(feats['final'])
            tone_idx    = torch.tensor(feats['tone'])
            radical_idx = torch.tensor(feats['radical'])
            stroke_idx  = torch.tensor(feats['stroke'])

            # 过Embedding层
            tok_pinyin  = self.pinyin_embed(pinyin_idx)   # (d_model,)
            tok_initial = self.initial_embed(initial_idx)   # (d_model,)
            tok_final   = self.final_embed(final_idx)       # (d_model,)
            tok_tone    = self.tone_embed(tone_idx)         # (d_model,)
            tok_radical = self.radical_embed(radical_idx)   # (d_model,)
            tok_stroke  = self.stroke_embed(stroke_idx)     # (d_model,)

            # 6个token聚合为1个类别原型向量（平均）
            proto = torch.stack([
                tok_pinyin, tok_initial, tok_final,
                tok_tone, tok_radical, tok_stroke
            ], dim=0).mean(dim=0)  # (d_model,)

            prototypes.append(proto)

        return torch.stack(prototypes, dim=0)  # (num_classes, d_model)

    def forward(self) -> torch.Tensor:
        """返回所有类别的语义原型向量: (num_classes, d_model)
        
        不需要传入任何标签！所有类别的语义信息在初始化时已预计算好。
        """
        return self.class_prototypes  # (num_classes, d_model)
