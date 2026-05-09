#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCATF + DTWTF + TF-NN 创新方法
- PCATF: 复数PCA (Complex PCA)
- DTWTF: 多变量DTW (Multivariate DTW)
- TF-NN: 频率加权K-NN (Frequency-Weighted K-NN)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
np.random.seed(42)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 輸出的數據類型就是原始的80*128的複數矩陣
# 複數域直接pca
def load_radar_data_complex(data_dir, labels_file, chars_file):
    data_dir = Path(data_dir)
    labels_df = pd.read_csv(data_dir / labels_file)
    chars_df = pd.read_csv(data_dir / chars_file)
    
    # 创建汉字到索引的映射
    char_to_idx = {char: idx for idx, char in enumerate(chars_df['char'])}
    idx_to_char = {idx: char for idx, char in enumerate(chars_df['char'])}
    
    height = 80
    width = 128
    
    print("加载复数雷达数据...")
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
        
        radar_data = np.zeros((height, width), dtype=complex)
        for i in range(height):
            for j in range(width):
                complex_str = df.iloc[i, j]
                radar_data[i, j] = complex(complex_str)
        
        all_features.append(radar_data)  # [80, 128], complex
        all_labels.append(label)
    
    all_features = np.array(all_features)  # [N, 80, 128], N是樣本數100
    all_labels = np.array(all_labels)  # [N]
    
    print(f"数据加载完成: {len(all_features)} 个样本")
    print(f"特征维度: {all_features.shape} (样本数, 时间步, 频率bin)")
    print(f"数据类型: {all_features.dtype}")
    
    return all_features, all_labels, idx_to_char


# 找到主成成分性價比最大的點，即最大解釋方差大於閾值，成分又要保證少
# 二階變化最大的那一個點
def determine_optimal_components(features_complex, variance_threshold=0.85, max_components=10):
    print("\n确定最优主成分数量...")
    N, T, F = features_complex.shape
    
    features_flat = features_complex.reshape(-1, F)  # [N*80, 128], complex
    
    # 中心化
    mean_complex = np.mean(features_flat, axis=0)  # [128], complex
    features_centered = features_flat - mean_complex  # [N*80, 128], complex
    
    # 计算复数协方差矩阵
    # 公式：C = 1/(n-1)*D*D^T，D是中心化后的数据
    # 且C是共轭对称矩阵H
    cov_complex = np.conj(features_centered.T) @ features_centered / len(features_centered)
    
    # 复数特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_complex)
    # eigenvalues: [128], 每個就是一個特徵值
    # eigenvectors: [128, 128], 每列是對應的特徵向量
    
    # 按特征值降序排列，特征值越大，对应的主成分越重要
    idx = np.argsort(eigenvalues)[::-1] # index是一個索引列表，然後按index排
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 計算至少需要多少個主成分才能達到85%
    total_variance = np.sum(eigenvalues) # 所有特徵值求和得縂方差
    explained_variance_ratio = eigenvalues / total_variance # 每個主成分的解釋方差比例
    cumulative_variance = np.cumsum(explained_variance_ratio) # 對解释方差比例做纍加和
    
    # 返回纍計解釋方差能夠大於閾值的第一個數的索引，從1開始
    n_components_variance = np.argmax(cumulative_variance >= variance_threshold) + 1
    # 和預定的最大成分數取最小值
    n_components_variance = min(n_components_variance, max_components)
    
    # 如果所有主成分都无法达到阈值，使用所有主成分
    if cumulative_variance[n_components_variance - 1] < variance_threshold:
        n_components_variance = min(len(cumulative_variance), max_components)
        print(f"  警告: 即使使用所有主成分，累计解释方差仅为 {cumulative_variance[n_components_variance-1]:.4f}")
    
    print(f"  步骤1: 达到{variance_threshold*100:.0f}%累计解释方差需要 {n_components_variance} 个主成分")
    print(f"        实际累计解释方差: {cumulative_variance[n_components_variance-1]:.4f} ({cumulative_variance[n_components_variance-1]*100:.2f}%)")
    
    # 在达到阈值的范围内（1到n_components_variance）使用肘部法则
    # 肘部法则，找解释方差下降速度突然变缓的点
    search_range = min(n_components_variance, max_components)
    
    if search_range > 1:
        # 一階段，上面計算的解釋方差，它們之間的差值絕對值
        variance_diff_abs = np.abs(np.diff(explained_variance_ratio[:search_range]))
        
        # 计算二阶差分
        if len(variance_diff_abs) > 1:
            second_diff = np.diff(variance_diff_abs)  # 差值再差值，下降速度的变化
            
            # 找到下降速度变化最大的点
            # 因爲這個點的變化速度最大，所以之後的變化速度全部減小，他就是突然變緩的點
            if len(second_diff) > 0:
                elbow_idx = np.argmax(second_diff)  # 找到变化最大的点
                n_components_elbow = elbow_idx + 2  # +2因为二阶差分
                n_components_elbow = min(n_components_elbow, search_range)
            else:
                n_components_elbow = search_range
        else:
            n_components_elbow = search_range
    else:
        n_components_elbow = 1
    
    # 确保肘部法则选择的数量不超过达到阈值所需的数量
    n_components_elbow = min(n_components_elbow, n_components_variance)
    
    # 验证：确保肘部法则选择的数量仍然满足累计解释方差阈值
    if cumulative_variance[n_components_elbow - 1] < variance_threshold:
        # 如果不满足，使用达到阈值所需的数量
        optimal_n_components = n_components_variance
        print(f"  步骤2: 肘部法则建议 {n_components_elbow} 个主成分")
        print(f"        但累计解释方差仅为 {cumulative_variance[n_components_elbow-1]:.4f}，不满足阈值")
        print(f"        因此使用 {n_components_variance} 个主成分以确保达到阈值")
    else:
        # 如果满足，使用肘部法则选择的数量
        optimal_n_components = n_components_elbow
        print(f"  步骤2: 在1-{n_components_variance}范围内，肘部法则建议 {n_components_elbow} 个主成分")
        print(f"        累计解释方差: {cumulative_variance[n_components_elbow-1]:.4f} ({cumulative_variance[n_components_elbow-1]*100:.2f}%)")
        print(f"        满足阈值要求，使用 {n_components_elbow} 个主成分")
    
    # 输出详细信息
    print(f"\n  前{max_components}个主成分的解释方差比例:")
    for i in range(min(max_components, len(explained_variance_ratio))):
        print(f"    主成分 {i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")
    
    print(f"  累计解释方差:")
    for i in range(min(max_components, len(cumulative_variance))):
        marker = " ← 达到阈值" if cumulative_variance[i] >= variance_threshold and i == n_components_variance - 1 else ""
        marker_elbow = " ← 肘部" if i == n_components_elbow - 1 else ""
        print(f"    前{i+1}个主成分: {cumulative_variance[i]:.4f} ({cumulative_variance[i]*100:.2f}%){marker}{marker_elbow}")
    
    print(f"\n  最终选择: {optimal_n_components} 个主成分")
    print(f"  累计解释方差: {cumulative_variance[optimal_n_components-1]:.4f} ({cumulative_variance[optimal_n_components-1]*100:.2f}%)")
    
    # 绘制解释方差比例图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, min(max_components+1, len(explained_variance_ratio)+1)), 
            explained_variance_ratio[:max_components])
    plt.axvline(x=n_components_variance, color='g', linestyle='--', 
                label=f'达到阈值: {n_components_variance}', linewidth=2)
    plt.axvline(x=optimal_n_components, color='r', linestyle='--', 
                label=f'最终选择: {optimal_n_components}', linewidth=2)
    plt.xlabel('主成分编号')
    plt.ylabel('解释方差比例')
    plt.title('各主成分解释方差比例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, min(max_components+1, len(cumulative_variance)+1)), 
             cumulative_variance[:max_components], 'o-', linewidth=2, markersize=6)
    plt.axhline(y=variance_threshold, color='g', linestyle='--', 
                label=f'阈值: {variance_threshold}', linewidth=2)
    plt.axvline(x=n_components_variance, color='g', linestyle='--', 
                label=f'达到阈值: {n_components_variance}', linewidth=2)
    plt.axvline(x=optimal_n_components, color='r', linestyle='--', 
                label=f'最终选择: {optimal_n_components}', linewidth=2)
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比例')
    plt.title('累计解释方差比例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'results' / 'pca_components_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  解释方差分析图已保存: pca_components_analysis.png")
    
    # 成分數量，各主成成分解釋方差，特徵向量，中心點
    return optimal_n_components, explained_variance_ratio, eigenvectors, mean_complex


def apply_complex_pca(features_complex, n_components=None, variance_threshold=0.85):
    print(f"\n应用复数PCA降维...")
    N, T, F = features_complex.shape  # N=样本数, T=80, F=128
    
    # 如果未指定主成分数，自动确定
    if n_components is None:
        n_components, explained_variance_ratio, eigenvectors, mean_complex = \
            determine_optimal_components(features_complex, variance_threshold)
    else:
        
        features_flat = features_complex.reshape(-1, F)  # [N*80, 128], complex
        mean_complex = np.mean(features_flat, axis=0)  # 中心點
        features_centered = features_flat - mean_complex # 中心化
        cov_complex = np.conj(features_centered.T) @ features_centered / len(features_centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_complex)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    
    # 選取前n_components个主成分特徵向量
    principal_components = eigenvectors[:, :n_components] # [128, n_components]
    
    print(f"  使用 {n_components} 个主成分")
    print(f"  累计解释方差: {np.sum(explained_variance_ratio[:n_components]):.4f} "
          f"({np.sum(explained_variance_ratio[:n_components])*100:.2f}%)")
    
    # 對k個主成分向量在每個bin上取平均
    frequency_weights = np.abs(principal_components).mean(axis=1)  # [128], 只有幅度real
    # k個向量對主成分貢獻程度
    # 暫時未被使用，且需要降維
    frequency_weights = frequency_weights / frequency_weights.sum()  # 归一化，[128]
    
    # 对每个样本应用PCA
    time_series_list = []
    for feat in features_complex:  # feat: [80, 128], complex
        feat_centered = feat - mean_complex  # 中心化
        # 投影到主成分空间
        projection = feat_centered @ principal_components  # [80, n_components], complex
        # 提取幅度，保留相位信息在PCA过程中
        projection_magnitude = np.abs(projection)  # [80, n_components], real
        time_series_list.append(projection_magnitude)
    
    time_series = np.array(time_series_list)  # [N, 80, n_components], real
    
    print(f"  时间序列形状: {time_series.shape}")
    
    # 中心化且投影到主成分數量后的數據，主成分個特徵向量，中心點，選取的特徵向量對主成分的貢獻程度
    return time_series, principal_components, mean_complex, frequency_weights


# 多了d是變量的數量，每個時間步不是一個變量了
def multivariate_dtw_distance(ts1, ts2):
    n, d1 = ts1.shape
    m, d2 = ts2.shape
    
    if d1 != d2:
        raise ValueError(f"变量维度不匹配: {d1} vs {d2}")
    
    # 创建初始矩阵
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # 计算DTW距离
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 多变量距离：欧氏距离
            cost = np.linalg.norm(ts1[i - 1] - ts2[j - 1])
            
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      
                dtw_matrix[i, j - 1],      
                dtw_matrix[i - 1, j - 1]   
            )
    
    return dtw_matrix[n, m]


class FrequencyWeightedKNN:
    """
    频率加权K-NN分类器
    使用多变量DTW距离，并根据频率权重进行加权投票
    """
    
    def __init__(self, k=3, frequency_weights=None):
        """
        参数:
            k: 最近邻数量
            frequency_weights: 前k個主要成分的特徵向量的貢獻程度
        """
        self.k = k
        self.frequency_weights = frequency_weights
        self.train_data = None
        self.train_labels = None
    
    def fit(self, X_train, y_train):
        """
        训练分类器
        
        参数:
            X_train: [N, T, d] 多变量时间序列
            y_train: [N] 标签
        """
        self.train_data = X_train
        self.train_labels = y_train
        print(f"频率加权K-NN分类器已训练")
        print(f"  训练样本数: {len(X_train)}")
        print(f"  K值: {self.k}")
        print(f"  时间序列维度: {X_train.shape[1:]} (时间步, 变量数)")
    
    def predict(self, X_test):
        predictions = []
        
        print(f"\n开始预测 {len(X_test)} 个测试样本...")
        start_time = time.time()
        
        # 遍歷測試樣本
        for i, test_ts in enumerate(X_test):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  已处理 {i + 1}/{len(X_test)} 个样本，耗时 {elapsed:.1f}s")
            
            # 遍歷訓練樣本及對應標簽
            distances = []
            for train_ts, train_label in zip(self.train_data, self.train_labels):
                distance = multivariate_dtw_distance(test_ts, train_ts)
                # 加入所有距離以及對應預測標簽(d1, label1), (d2, label2)
                distances.append((distance, train_label))
            
            # 按每個元素第0個元素排，所以距離最小的在前面
            distances.sort(key=lambda x: x[0])
            # k-近鄰，取前k個
            k_nearest = distances[:self.k]
            
            # 對前k個如果有重複的進行叠加
            # 如果提供了频率权重，可以根据距离和权重进行加权
            # 这里简化处理：直接使用距离的倒数作为权重，距離越小權重越大
            votes = {}
            for distance, label in k_nearest:
                weight = 1.0 / (1.0 + distance)
                if label not in votes: # 初始化一下
                    votes[label] = 0
                votes[label] += weight
            
            # 选择得票最多的标签
            predicted_label = max(votes, key=votes.get)
            predictions.append(predicted_label)
        
        return np.array(predictions)
    
    def predict_single(self, test_ts):
        """预测单个样本"""
        distances = []
        for train_ts, train_label in zip(self.train_data, self.train_labels):
            distance = multivariate_dtw_distance(test_ts, train_ts)
            distances.append((distance, train_label))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        votes = {}
        for distance, label in k_nearest:
            weight = 1.0 / (1.0 + distance)
            if label not in votes:
                votes[label] = 0
            votes[label] += weight
        
        return max(votes, key=votes.get)


def calculate_metrics(y_true, y_pred, idx_to_char):
    """计算评估指标"""
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # 获取唯一标签
    unique_labels = sorted(set(y_true) | set(y_pred))
    char_names = [idx_to_char[i] for i in unique_labels]
    
    # 分类报告
    report = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=char_names,
        zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    return accuracy, report, cm, unique_labels, char_names


def main():
    """主函数"""
    print("=" * 70)
    print("PCATF + DTWTF + TF-NN 创新方法")
    print("=" * 70)
    print("PCATF: 复数PCA (Complex PCA)")
    print("DTWTF: 多变量DTW (Multivariate DTW)")
    print("TF-NN: 频率加权K-NN (Frequency-Weighted K-NN)")
    print("=" * 70)
    
    # 检查数据集
    data_dir = PROJECT_ROOT / "chinese_radar_dataset"
    if not data_dir.exists():
        print(f"数据集目录 {data_dir} 不存在！")
        return
    
    # 1. 加载数据（保持复数格式）
    print("\n1. 加载复数雷达数据...")
    features_complex, labels, idx_to_char = load_radar_data_complex(
        data_dir, 'labels.csv', 'characters.csv'
    )
    
    num_classes = len(set(labels))
    print(f"   类别数量: {num_classes}")
    
    # 2. 应用复数PCA
    print("\n2. 应用复数PCA降维...")
    # 自动确定主成分数量（累计解释方差>=85%）
    time_series, principal_components, mean_complex, frequency_weights = \
        apply_complex_pca(features_complex, n_components=None, variance_threshold=0.85)
    
    print(f"   时间序列形状: {time_series.shape} (样本数, 时间步, 主成分数)")
    
    # 3. 划分训练集和验证集
    print("\n3. 划分训练集和验证集...")
    X_train, X_test, y_train, y_test = train_test_split(
        time_series, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   训练集: {len(X_train)} 个样本")
    print(f"   验证集: {len(X_test)} 个样本")
    
    # 4. 训练频率加权K-NN分类器
    print("\n4. 训练频率加权K-NN分类器...")
    k = 3  # K值，可以根据需要调整
    classifier = FrequencyWeightedKNN(k=k, frequency_weights=frequency_weights)
    classifier.fit(X_train, y_train)
    
    # 5. 预测
    print("\n5. 在验证集上进行预测...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    elapsed_time = time.time() - start_time
    
    # 6. 计算评估指标
    print("\n6. 计算评估指标...")
    accuracy, report, cm, unique_labels, char_names = calculate_metrics(
        y_test, y_pred, idx_to_char
    )
    
    # 7. 输出结果
    print("\n" + "=" * 70)
    print("结果:")
    print(f"  预测耗时: {elapsed_time:.2f} 秒")
    print(f"  准确率: {accuracy:.2f}%")
    print("\n分类报告:")
    print(report)
    print("=" * 70)
    
    # 8. 绘制混淆矩阵
    if len(unique_labels) <= 50:  # 如果类别不太多，绘制混淆矩阵
        plt.figure(figsize=(max(12, len(unique_labels)*0.5), max(10, len(unique_labels)*0.5)))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=char_names, yticklabels=char_names)
        plt.title(f'混淆矩阵 (准确率: {accuracy:.2f}%)')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'results' / 'confusion_matrix_tf_innovative.png', dpi=300, bbox_inches='tight')
        print(f"\n混淆矩阵已保存: confusion_matrix_tf_innovative.png")
    else:
        print(f"\n类别数量过多({len(unique_labels)})，跳过混淆矩阵绘制")
    
    # 9. 保存结果
    result_filename = PROJECT_ROOT / 'results' / 'tf_innovative_result.txt'
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("PCATF + DTWTF + TF-NN 创新方法结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"准确率: {accuracy:.2f}%\n")
        f.write(f"预测耗时: {elapsed_time:.2f} 秒\n")
        f.write(f"K值: {k}\n")
        f.write(f"主成分数: {time_series.shape[2]}\n")
        f.write(f"时间序列维度: {time_series.shape[1:]} (时间步, 主成分数)\n\n")
        f.write("分类报告:\n")
        f.write("-" * 70 + "\n")
        f.write(report)
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n结果已保存到: {result_filename}")


if __name__ == "__main__":
    main()

