#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GRU的中文雷达语音分类模型
"""

import torch
import torch.nn as nn
import numpy as np


class RadarGRU(nn.Module):
    """基于GRU的雷达数据分类模型"""
    
    def __init__(self, input_size=256, hidden_size=64, num_layers=2, num_classes=416, dropout=0.3):
        """
        初始化GRU模型
        
        Args:
            input_size: 输入特征维度（128个bins × 2通道 = 256）
            hidden_size: 隐藏层大小
            num_layers: GRU层数
            num_classes: 分类数量（汉字数量）
            dropout: Dropout概率
        """
        super(RadarGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,  # 256 = 128 bins × 2 channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, channels, height, width]
               - channels=2: 实部和虚部
               - height=80: 时间步数
               - width=128: 每个时间步的特征数（频率bins）
        
        Returns:
            分类结果 [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # 重塑输入：保留时间维度，合并通道和特征维度
        # [batch_size, 2, 80, 128] -> [batch_size, 80, 2, 128] -> [batch_size, 80, 256]
        x = x.permute(0, 2, 1, 3)  # [batch_size, 80, 2, 128]
        x = x.reshape(batch_size, 80, -1)  # [batch_size, 80, 256]
        
        # GRU处理：沿时间维度（80个时间步）循环
        gru_out, _ = self.gru(x)  # [batch_size, 80, hidden_size*2]
        
        # 全局平均池化：聚合所有时间步信息
        pooled = torch.mean(gru_out, dim=1)  # [batch_size, hidden_size*2]
        
        # 分类
        output = self.classifier(pooled)
        
        return output