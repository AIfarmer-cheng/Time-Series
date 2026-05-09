# PCATF + DTWTF + TF-NN 创新方法说明

## 方法概述

### PCATF (复数PCA)
- **原理**: 直接对复数数据做PCA，保留相位信息
- **优势**: 更好地利用复数数据的特性，保留幅度和相位关系

### DTWTF (多变量DTW)
- **原理**: 对多变量时间序列（多个主成分）同时做DTW对齐
- **优势**: 利用更多频率信息，比单变量DTW更准确

### TF-NN (频率加权K-NN)
- **原理**: K个最近邻加权投票，根据频率重要性加权
- **优势**: 更稳健的分类，利用频率权重信息

## 主成分数量确定方法

### 方法1: 累计解释方差比例（主要方法）

**原理**: 选择累计解释方差达到阈值（如85%）的主成分数量

```python
# 在代码中：
variance_threshold = 0.85  # 85%的累计解释方差
n_components = determine_optimal_components(features, variance_threshold=0.85)
```

**判断标准**:
- **85%**: 保留大部分信息，适合一般情况（推荐）
- **90%**: 保留更多信息，但维度可能较高
- **80%**: 更激进的降维，可能丢失部分信息

**输出示例**:
```
前10个主成分的解释方差比例:
  主成分 1: 0.4523 (45.23%)
  主成分 2: 0.2134 (21.34%)
  主成分 3: 0.0987 (9.87%)
  ...

累计解释方差:
  前1个主成分: 0.4523 (45.23%)
  前2个主成分: 0.6657 (66.57%)
  前3个主成分: 0.7644 (76.44%)
  前4个主成分: 0.8234 (82.34%)
  前5个主成分: 0.8712 (87.12%)  ← 达到85%阈值

选择的主成分数: 5
```

### 方法2: 肘部法则（辅助方法）

**原理**: 找到解释方差下降最快的点（"肘部"）

**判断标准**:
- 观察解释方差比例的下降趋势
- 选择下降速度突然变缓的点

**示例**:
```
主成分 1: 45.23%  (下降: -)
主成分 2: 21.34%  (下降: 23.89%)
主成分 3: 9.87%   (下降: 11.47%)  ← 下降速度变缓
主成分 4: 5.67%   (下降: 4.20%)
```

### 方法3: 交叉验证（可选，需要额外实现）

**原理**: 在不同主成分数量下进行交叉验证，选择准确率最高的

**实现思路**:
```python
def cross_validate_components(features, labels, max_components=10):
    best_n = 1
    best_acc = 0
    
    for n in range(1, max_components + 1):
        # 应用PCA
        time_series, _, _, _ = apply_complex_pca(features, n_components=n)
        
        # 交叉验证
        acc = cross_val_score(classifier, time_series, labels, cv=5).mean()
        
        if acc > best_acc:
            best_acc = acc
            best_n = n
    
    return best_n
```

## 参数调整建议

### 1. 主成分数量 (`n_components`)

**自动确定（推荐）**:
```python
time_series, _, _, _ = apply_complex_pca(
    features_complex, 
    n_components=None,  # 自动确定
    variance_threshold=0.85  # 85%累计解释方差
)
```

**手动指定**:
```python
time_series, _, _, _ = apply_complex_pca(
    features_complex, 
    n_components=5  # 手动指定5个主成分
)
```

**选择建议**:
- **1-3个主成分**: 高度降维，适合简单模式
- **3-5个主成分**: 平衡降维和信息保留（推荐）
- **5-10个主成分**: 保留更多信息，但计算量增加
- **>10个主成分**: 通常不推荐，维度太高

### 2. K值（K-NN中的K）

```python
k = 3  # 默认值
classifier = FrequencyWeightedKNN(k=k, frequency_weights=frequency_weights)
```

**选择建议**:
- **K=1**: 1-NN，最简单但可能不稳定
- **K=3**: 平衡稳定性和准确性（推荐）
- **K=5**: 更稳健，但计算量增加
- **K=7或更大**: 适合噪声较大的数据

**经验法则**: K通常选择奇数，避免平票

### 3. 累计解释方差阈值 (`variance_threshold`)

```python
variance_threshold = 0.85  # 85%
```

**选择建议**:
- **0.80 (80%)**: 更激进的降维
- **0.85 (85%)**: 平衡（推荐）
- **0.90 (90%)**: 保留更多信息
- **0.95 (95%)**: 几乎保留所有信息

## 运行方法

### 基本运行
```bash
python train_tf_innovative.py
```

### 修改参数
编辑 `train_tf_innovative.py` 中的 `main()` 函数：

```python
# 修改主成分数量确定方法
time_series, _, _, _ = apply_complex_pca(
    features_complex, 
    n_components=None,  # 或指定数字，如 n_components=5
    variance_threshold=0.85  # 调整阈值
)

# 修改K值
k = 3  # 改为其他值，如 k=5
classifier = FrequencyWeightedKNN(k=k, frequency_weights=frequency_weights)
```

## 输出文件

1. **pca_components_analysis.png**: 主成分分析图
   - 左图: 各主成分解释方差比例
   - 右图: 累计解释方差比例曲线

2. **confusion_matrix_tf_innovative.png**: 混淆矩阵（如果类别数≤50）

3. **tf_innovative_result.txt**: 详细结果报告

## 与原始方法对比

| 方法 | PCA类型 | DTW类型 | NN类型 | 主成分数 |
|------|---------|---------|--------|---------|
| **原始方法** | 实数PCA（分离实部虚部） | 单变量DTW | 1-NN | 1（固定） |
| **创新方法** | 复数PCA | 多变量DTW | 频率加权K-NN | 自动确定（1-10） |

## 性能优化建议

1. **如果数据量大**: 
   - 减少主成分数量（如3-5个）
   - 使用较小的K值（如K=3）

2. **如果准确率不够**:
   - 增加主成分数量（如5-7个）
   - 提高累计解释方差阈值（如0.90）
   - 增加K值（如K=5）

3. **如果计算太慢**:
   - 减少主成分数量
   - 使用较小的K值
   - 考虑使用近似DTW算法

## 常见问题

### Q1: 主成分数量应该选多少？
**A**: 建议使用自动确定方法（`n_components=None`），系统会根据累计解释方差自动选择。通常3-5个主成分比较合适。

### Q2: 累计解释方差阈值怎么选？
**A**: 
- 85%是较好的平衡点（推荐）
- 如果数据噪声大，可以降低到80%
- 如果需要保留更多信息，可以提高到90%

### Q3: K值怎么选？
**A**: 
- K=3是较好的起点（推荐）
- 如果数据噪声大，可以增加到K=5或7
- 如果数据很干净，K=1也可以

### Q4: 为什么使用复数PCA而不是分离实部虚部？
**A**: 复数PCA保留了相位信息，能更好地利用复数数据的特性。分离实部虚部会丢失相位关系。

### Q5: 多变量DTW比单变量DTW好在哪里？
**A**: 多变量DTW同时对齐多个主成分，能利用更多频率信息，通常比单变量DTW更准确。

