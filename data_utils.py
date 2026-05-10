#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达数据集加载工具
==================
功能：加载中文雷达无声语音识别数据集，将复数值CSV文件转换为PyTorch张量。

数据格式：
- 每个CSV文件包含80行×128列的复数数据（格式：a+bj）
- 拆分实部和虚部为两个通道 → 最终形状 (2, 80, 128)
- labels.csv: filename, char, char_index, sample_index
- characters.csv: char, char_index

作者：硕士毕业设计 - 无声语音识别
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class RadarDataset(Dataset):
    """雷达复数数据集加载器
    
    每个样本是一个80帧×128频点的雷达时频谱（复数IQ数据）。
    加载时自动将复数的实部和虚部分离为两个独立通道。
    """

    def __init__(
        self,
        data_dir: str,
        labels_file: str = 'labels.csv',
        characters_file: str = 'characters.csv',
        use_magnitude_phase: bool = False,
    ):
        """
        Args:
            data_dir: 数据集根目录路径（包含data/子目录）
            labels_file: 标签CSV文件名
            characters_file: 字符列表CSV文件名
            use_magnitude_phase: 是否使用幅度-相位表示（默认False，使用实部-虚部）
        """
        self.data_dir = Path(data_dir)
        self.data_subdir = self.data_dir / 'data'
        self.use_magnitude_phase = use_magnitude_phase

        # 加载标签和字符映射
        self.labels_df = pd.read_csv(self.data_dir / labels_file, encoding='utf-8-sig')
        self.chars_df = pd.read_csv(self.data_dir / characters_file, encoding='utf-8-sig')

        # 构建字符索引到字符名称的映射
        self.char_mapping = dict(zip(self.chars_df['char_index'], self.chars_df['char']))
        # 构建字符名称到索引的映射
        self.char_to_idx = dict(zip(self.chars_df['char'], self.chars_df['char_index']))

        # 预验证所有文件存在
        self._validate_files()

    def _validate_files(self):
        """验证所有标签记录对应的CSV文件是否存在"""
        missing = []
        for _, row in self.labels_df.iterrows():
            fpath = self.data_subdir / row['filename']
            if not fpath.exists():
                missing.append(row['filename'])
        if missing:
            raise FileNotFoundError(f"缺少 {len(missing)} 个数据文件: {missing[:5]}...")

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """加载单个样本
        
        Returns:
            data: 张量，形状 (2, 80, 128) - [实部, 虚部] 或 [幅度, 相位]
            label: 整数标签 (0-99)
        """
        row = self.labels_df.iloc[idx]
        filepath = self.data_subdir / row['filename']
        label = int(row['char_index'])

        # 读取CSV：每行是逗号分隔的复数字符串
        complex_data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 解析形如 "a+bj" 的复数字符串
                values = [complex(v) for v in line.split(',') if v]
                complex_data.append(values)

        # 转换为numpy数组 (80, 128)
        arr = np.array(complex_data, dtype=np.complex64)

        if self.use_magnitude_phase:
            # 幅度-相位表示：更适合某些模型理解信号强度
            magnitude = np.abs(arr)
            phase = np.angle(arr)
            tensor = torch.from_numpy(np.stack([magnitude, phase], axis=0))
        else:
            # 实部-虚部表示：保留完整复数信息
            real = np.real(arr)
            imag = np.imag(arr)
            tensor = torch.from_numpy(np.stack([real, imag], axis=0))

        return tensor.float(), label

    def get_char_name(self, idx: int) -> str:
        """根据类别索引获取汉字名称"""
        return self.char_mapping.get(idx, f'Unknown_{idx}')

    def get_char_index(self, char: str) -> Optional[int]:
        """根据汉字获取类别索引"""
        return self.char_to_idx.get(char)

    @property
    def num_classes(self) -> int:
        """类别总数"""
        return len(self.chars_df)

    @property
    def char_list(self) -> list:
        """所有汉字列表（按索引顺序）"""
        return self.chars_df['char'].tolist()
