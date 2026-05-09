#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用PCA处理时频雷达数据, 转换为单变量时间序列, 并用DTW+1-NN分类
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time

# 设置随机种子
np.random.seed(42)


# 计算两个时间序列之间的DTW距离(动态规划)
# 时间复杂度: O(n*m)
# 空间复杂度: O(n*m)
# ts1: 时间序列1 [n]
# ts2: 时间序列2 [m]
# 返回: DTW距离
def dtw_distance(ts1, ts2):
    n = len(ts1)
    m = len(ts2)
    
    # 创建距离矩阵(防止边界)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    # 初始化(0,0)位置为0
    dtw_matrix[0, 0] = 0
    
    # 计算DTW距离
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(ts1[i - 1] - ts2[j - 1]) # 数值相减+矩阵左下方最小值
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      
                dtw_matrix[i, j - 1],      
                dtw_matrix[i - 1, j - 1]   
            )
    
    return dtw_matrix[n, m] # 只返回最後一個值，可以理解爲兩條序列的誤差值

# 返回一個80*256簡單分離拼接的矩陣，和對應的label
def load_radar_data(data_dir, labels_file, chars_file):
    data_dir = Path(data_dir)
    labels_df = pd.read_csv(data_dir / labels_file) # 路徑+文件名
    chars_df = pd.read_csv(data_dir / chars_file) # 一個字對應一個數字，從0-99
    
    # 创建汉字到索引的映射
    char_to_idx = {char: idx for idx, char in enumerate(chars_df['char'])}
    idx_to_char = {idx: char for idx, char in enumerate(chars_df['char'])}
    
    height = 80
    width = 128
    
    print("加载数据并提取实部和虚部...")
    all_features = []
    all_labels = []
    
    for idx in range(len(labels_df)):
        row = labels_df.iloc[idx]
        filename = row['filename']
        char = row['char']
        label = char_to_idx[char]
        
        # 加载复数数据
        filepath = data_dir / "data" / filename
        df = pd.read_csv(filepath, header=None)
        
        radar_data = np.zeros((height, width), dtype=complex) # 80x128的全0矩陣
        for i in range(height):
            for j in range(width):
                complex_str = df.iloc[i, j] # 获取第i行第j列的值
                radar_data[i, j] = complex(complex_str)  # 将字符串转换为复数，再放進去
        
        # 提取实部和虚部分開來的兩個矩陣
        real_part = radar_data.real  # [80, 128] 僅实部
        imag_part = radar_data.imag  # [80, 128] 僅虚部
        
        # 沿列（频率维度）拼接，簡單的拼接而已
        features = np.concatenate([real_part, imag_part], axis=1)  # [80, 256]
        all_features.append(features)
        all_labels.append(label)
    
    all_features = np.array(all_features)  # [N, 80, 256]，N是樣本數100
    all_labels = np.array(all_labels)  # [N]
    
    print(f"数据加载完成: {len(all_features)} 个样本")
    print(f"特征维度: {all_features.shape} (样本数, 时间步, 频率特征)")
    
    return all_features, all_labels, idx_to_char

# pca降维
def apply_pca(features):
    print("应用PCA降维...")
    # 重塑为 [N*80, 256] 用于PCA拟合 (256 = 128实部 + 128虚部)
    features_flat = features.reshape(-1, 256)
    
    pca = PCA(n_components=1) # 創建一個PCA對象，告訴它降到1維
    pca.fit(features_flat) # 尋找主成分方向
    
    print(f"PCA拟合完成, 解释方差比例: {pca.explained_variance_ratio_[0]:.4f}")
    
    # 对每个样本应用PCA
    time_series_list = []
    for feat in features:
        ts = pca.transform(feat)  # [80, 1]
        time_series_list.append(ts.flatten())  # [80]，變成1行80列
    
    time_series = np.array(time_series_list)  # [N, 80]
    
    return time_series, pca


# 暴力動態規劃
class OneNNClassifier:
    def __init__(self):
        self.train_data = None
        self.train_labels = None
    
    def fit(self, X_train, y_train):
        
        self.train_data = X_train
        self.train_labels = y_train
        print(f"1-NN分类器已训练,存储了 {len(X_train)} 个训练样本")
    
    def predict(self, X_test):
        predictions = []
        print(f"开始预测 {len(X_test)} 个测试样本...")
        start_time = time.time()
        for i, test_ts in enumerate(X_test):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  已处理 {i + 1}/{len(X_test)} 个样本，耗时 {elapsed:.1f}s")
            
            # 计算与所有训练样本的DTW距离
            min_distance = float('inf') # 初始化最小距离为无穷大
            nearest_label = None # 初始化最近邻标签为None
            
            for train_ts, train_label in zip(self.train_data, self.train_labels): # zip配对两个列表，再比较
                distance = dtw_distance(test_ts, train_ts) # 直接拿測試樣本和訓練樣本計算距離
                if distance < min_distance:
                    min_distance = distance
                    nearest_label = train_label # 暴力搜索找到最小誤差值序列，將其標簽加入
            
            predictions.append(nearest_label)
        
        return np.array(predictions)
    
    def predict_single(self, test_ts):
        
        min_distance = float('inf')
        nearest_label = None
        
        for train_ts, train_label in zip(self.train_data, self.train_labels):
            distance = dtw_distance(test_ts, train_ts)
            if distance < min_distance:
                min_distance = distance
                nearest_label = train_label
        
        return nearest_label


def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = 100.0 * correct / total
    return accuracy, correct, total


def main():
    """主函数"""
    print("=" * 70)
    print("PCA + DTW + 1-NN 雷达数据分类")
    print("=" * 70)
    
    # 检查数据集
    data_dir = PROJECT_ROOT / "chinese_radar_dataset"
    if not data_dir.exists():
        print(f"数据集目录 {data_dir} 不存在！")
        return
    
    # 加载数据
    print("\n1. 加载数据...")
    features, labels, idx_to_char = load_radar_data(
        data_dir, 'labels.csv', 'characters.csv'
    )
    
    num_classes = len(set(labels))
    print(f"   类别数量: {num_classes}")
    
    # 应用PCA
    print("\n2. 应用PCA降维（实部+虚部）...")
    time_series, pca = apply_pca(features)
    print(f"   时间序列形状: {time_series.shape}")
    
    # 划分训练集和验证集
    print("\n3. 划分训练集和验证集...")
    # X是輸入y是標簽
    X_train, X_test, y_train, y_test = train_test_split(
        time_series, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   训练集: {len(X_train)} 个样本")
    print(f"   验证集: {len(X_test)} 个样本")
    
    # 训练1-NN分类器
    print("\n4. 训练1-NN分类器...")
    classifier = OneNNClassifier()
    classifier.fit(X_train, y_train)
    
    # 预测
    print("\n5. 在验证集上进行预测...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    elapsed_time = time.time() - start_time
    
    # 计算准确率
    print("\n6. 计算准确率...")
    accuracy, correct, total = calculate_accuracy(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("结果:")
    print(f"  预测耗时: {elapsed_time:.2f} 秒")
    print(f"  正确预测: {correct}/{total}")
    print(f"  准确率: {accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
