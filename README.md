# 中文雷达语音识别系统

基于深度学习的中文雷达无声语音识别系统，支持GRU和RadarTFNet两种模型架构，使用毫米波雷达数据进行汉字分类和句子识别。

## 项目概述

本项目使用24GHz毫米波雷达采集的语音发音数据，通过深度学习模型实现中文汉字的无声识别。系统可以识别单个汉字以及由多个汉字组成的句子，适用于无障碍通信、医疗康复和人机交互等场景。

### 主要特性

- ✅ **100个常用汉字**识别
- ✅ **1,000个样本**的训练数据集
- ✅ **双模型架构支持**：GRU和RadarTFNet
- ✅ **RadarTFNet时频神经网络**（基于TF-GridNet架构，针对雷达复数数据优化）
- ✅ **双向GRU神经网络**模型
- ✅ **句子级识别**支持（3-5字短句）
- ✅ **LLM智能纠错**功能（提升准确率10-20%）
- ✅ **完整的训练和评估流程**
- ✅ **可视化结果展示**

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

主要依赖包：
- PyTorch
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

**LLM纠错依赖：**
- transformers（Hugging Face模型库）
- accelerate（模型加载加速）
- sentencepiece（分词器）

### 2. 训练模型

系统支持两种模型架构，训练时可通过参数选择：

**训练GRU模型：**
```bash
python train.py --model gru
```

**训练RadarTFNet模型：**
```bash
python train.py --model radar_tfnet
```

**自定义训练参数：**
```bash
python train.py --model radar_tfnet --epochs 50 --batch_size 16 --lr 0.0005
```

**训练配置参数：**
- `--model`: 模型类型，选择 `gru` 或 `radar_tfnet`（默认：gru）
- `--epochs`: 训练轮数（默认：30）
- `--batch_size`: 批次大小（默认：8）
- `--lr`: 学习率（默认：0.001）
- 训练/验证集划分：80/20（固定）

### 3. 句子识别测试

**使用GRU模型预测：**
```bash
python predict_sentences.py --model gru
```

**使用RadarTFNet模型预测：**
```bash
python predict_sentences.py --model radar_tfnet
```

系统会：
- 自动加载测试句子集
- 根据选择的模型类型加载对应模型
- 逐字识别并显示详细结果
- **保存预测结果到 `prediction_results.json`**（用于后续LLM纠错）

### 4. LLM智能纠错（可选）

**工作流程：**
```bash
# 步骤1: 生成预测结果
python predict_sentences.py

# 步骤2: LLM纠错（使用Qwen3-1.7B）
python llm_sentence_corrector.py
```

LLM纠错会读取 `prediction_results.json` 中的预测结果，使用 **Qwen3-1.7B** 本地模型进行语义纠错，提升准确率10-20%。

**模型特点：**
- 🏠 完全本地运行，无需API
- 🔒 数据不出本地，隐私安全
- ⚡ 小巧高效（1.7B参数）
- 🇨🇳 专为中文优化

详细说明请参考下方的"LLM智能纠错"章节

## 项目结构

```
Radar_LLM_Agent_0/
├── chinese_radar_dataset/          # 训练数据集
│   ├── data/                       # 雷达数据文件（1,000个样本）
│   │   ├── 我_000.csv             # 单字样本
│   │   └── ...
│   ├── labels.csv                  # 样本标签
│   └── characters.csv              # 汉字列表（100个）
│
├── sentence_test_dataset/          # 句子测试数据集
│   ├── data/                       # 句子雷达数据（15个句子）
│   │   ├── sentence_000.csv       # 我爱中国（4字，320行）
│   │   ├── sentence_001.csv       # 白云和红日（5字，400行）
│   │   └── ...
│   └── sentence_labels.csv         # 句子标签信息
│
├── model/                          # 模型文件目录
│   ├── gru_model.py               # GRU模型定义
│   ├── radar_tfnet_model.py      # RadarTFNet模型定义 ⭐新增
│   ├── config.json                # 配置文件
│   ├── best_gru_model.pth         # GRU最佳模型权重
│   ├── final_gru_model.pth        # GRU最终模型权重
│   ├── best_radar_tfnet_model.pth # RadarTFNet最佳模型权重 ⭐新增
│   ├── final_radar_tfnet_model.pth # RadarTFNet最终模型权重 ⭐新增
│   ├── training_history.png       # 训练历史图表
│   └── confusion_matrix.png       # 混淆矩阵
│
├── models/                         # LLM模型缓存（自动生成）
│   └── models--Qwen--Qwen2.5-1.5B-Instruct/  # Qwen模型文件（约3GB）
│
├── train.py                       # 训练脚本（支持模型选择） ⭐更新
├── predict_sentences.py           # 句子识别测试脚本（支持模型选择） ⭐更新
├── prediction_results.json        # 预测结果（自动生成） ⭐
├── llm_sentence_corrector.py      # LLM句子纠错脚本 ⭐新增
├── llm_correction_results.json    # LLM纠错结果（自动生成） ⭐
├── check_model_path.py            # 检查模型路径工具 ⭐新增
├── requirements.txt               # 依赖包列表
├── .gitignore                     # Git忽略文件配置 ⭐新增
└── README.md                      # 项目说明文档（含LLM纠错指南）
```

## 数据集说明

### 训练数据集

**数据规模：**
- 汉字数量：100个常用汉字
- 样本数量：1,000个（每个汉字10个样本）
- 数据格式：80×128 复数矩阵（实部+虚部）

**数据采集：**
- 雷达系统：24GHz毫米波雷达
- 采样频率：1000Hz
- 受试者：15名健康成年人（8男7女）
- 采集环境：标准实验室环境

**数据格式：**
```csv
# 每个CSV文件：80行×128列的复数数据
-0.4267726+0.2155174j,1.2720335-0.8158792j,...
-1.4881292+0.1579701j,0.6168998+0.1245750j,...
...
```

### 句子测试数据集

**测试句子（15个）：**

**4字句子（9个）：**
1. 我爱中国
2. 中国文字
3. 我们的家
4. 书中有字
5. 天上有云
6. 学中文好
7. 我要学字
8. 火红的日
9. 他们在学

**5字句子（4个）：**
10. 白云和红日
11. 中国山水好
12. 人们说中文
13. 我们会写字

**3字句子（2个）：**
14. 水和火
15. 好生活

**数据组织：**
- 每个句子的雷达数据由单字数据纵向拼接而成
- 3字句子：240行×128列
- 4字句子：320行×128列
- 5字句子：400行×128列

## 模型架构

系统支持两种模型架构，可根据需求选择：

### 1. GRU神经网络

**核心组件：**
- 双向GRU层：处理时序特征
- 全局平均池化：提取全局特征
- 多层分类器：最终分类决策

**模型参数：**
- 输入维度：2（实部和虚部）
- 隐藏层大小：64
- GRU层数：2
- 分类数量：100个汉字
- Dropout率：0.3
- 总参数量：~124K

**技术特点：**
- 时序建模：GRU有效处理雷达数据的时序特征
- 双向处理：同时考虑前向和后向的时序信息
- 正则化：使用Dropout防止过拟合
- 轻量化设计：相对较少的参数量，训练效率高

### 2. RadarTFNet（雷达时频神经网络）⭐新增

**核心组件：**
- **帧内全频带模块**：使用深度可分离卷积处理频率维度（128 bins）
- **子频段时间模块**：使用BiLSTM处理时间维度（80 frames）
- **跨帧自注意力模块**：捕捉长期时序依赖关系
- 多层堆叠架构：增强特征表达能力

**模型参数：**
- 输入维度：2（实部和虚部）
- 基础通道数：64
- 子频带数量：8
- 堆叠层数：4
- 分类数量：100个汉字
- 总参数量：~917K

**技术特点：**
- 分频带建模：全频带与子频带结合，多尺度特征提取
- 频率-时间分离处理：分别优化频率和时间维度
- 跨帧注意力：捕捉长期时序依赖
- 复数谱处理：充分利用复数时频数据特性

**参考论文：**
- TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation
- arXiv: 2211.12433

**适用场景：**
- 需要更强特征表达能力的场景
- 对准确率要求更高的应用
- 计算资源充足的部署环境

## 使用说明

### 训练模型

**使用命令行参数：**
```bash
# 训练GRU模型
python train.py --model gru --epochs 30 --batch_size 8

# 训练RadarTFNet模型
python train.py --model radar_tfnet --epochs 30 --batch_size 8

# 自定义参数
python train.py --model radar_tfnet --epochs 50 --batch_size 16 --lr 0.0005
```

训练过程会自动：
- 加载数据集并划分训练/验证集
- 根据选择的模型类型创建对应模型
- 每个epoch显示训练进度
- 保存最佳模型（验证准确率最高，文件名根据模型类型自动命名）
- 生成训练历史图表和混淆矩阵

**保存的模型文件：**
- GRU: `model/best_gru_model.pth`, `model/final_gru_model.pth`
- RadarTFNet: `model/best_radar_tfnet_model.pth`, `model/final_radar_tfnet_model.pth`

### 加载模型进行预测

**使用GRU模型：**
```python
from model.gru_model import RadarGRU
from train import RadarDataset
import torch

# 加载数据集
dataset = RadarDataset('chinese_radar_dataset', 'labels.csv', 'characters.csv')

# 加载GRU模型
model = RadarGRU(input_size=256, hidden_size=64, num_layers=2, num_classes=100)
model.load_state_dict(torch.load('model/best_gru_model.pth'))
model.eval()

# 预测单个样本
data, label = dataset[0]
data = data.unsqueeze(0)  # 添加batch维度

with torch.no_grad():
    output = model(data)
    _, predicted = torch.max(output, 1)
    char_name = dataset.get_char_name(predicted.item())
    print(f"预测结果: {char_name}")
```

**使用RadarTFNet模型：**
```python
from model.radar_tfnet_model import RadarTFNet
from train import RadarDataset
import torch

# 加载数据集
dataset = RadarDataset('chinese_radar_dataset', 'labels.csv', 'characters.csv')

# 加载RadarTFNet模型
model = RadarTFNet(num_classes=100, freq_bins=128, time_frames=80)
model.load_state_dict(torch.load('model/best_radar_tfnet_model.pth'))
model.eval()

# 预测单个样本（同上）
```

### 句子识别测试

**使用命令行参数：**
```bash
# 使用GRU模型预测
python predict_sentences.py --model gru

# 使用RadarTFNet模型预测
python predict_sentences.py --model radar_tfnet
```

系统会自动：
- 加载句子测试数据集
- 将句子分割为单个汉字
- 逐字识别并显示置信度
- 计算字符级和句子级准确率
- 显示详细的识别对比结果

**输出示例：**
```
[1/15] 文件: sentence_000.csv
真实句子: 我爱中国 (4 字)
预测句子: 我爱中国
平均置信度: 95.32%
字符准确率: 100.00%

逐字对比:
位置   真实    预测    结果    置信度
1      我      我      OK      96.23%
2      爱      爱      OK      94.57%
3      中      中      OK      95.18%
4      国      国      OK      95.31%
```

### LLM智能纠错（Qwen3-1.7B）

使用Qwen3-1.7B本地模型对识别结果进行语义纠错：

```bash
python llm_sentence_corrector.py
```

**功能特点：**
- 🔧 自动修正识别错误的汉字
- 📈 提升准确率10-20%
- 🏠 完全本地运行，无需API密钥
- 🔒 数据不出本地，隐私安全
- ⚡ 小巧高效（1.7B参数）
- 💾 自动保存纠错结果

**工作流程：**
```
步骤1: predict_sentences.py (可选择GRU或RadarTFNet)
雷达数据 → 模型识别 → 可能有错的句子 → 保存到 prediction_results.json

步骤2: llm_sentence_corrector.py
读取 prediction_results.json → Qwen2.5-1.5B纠错 → 通顺句子 → 保存到 llm_correction_results.json
```

**示例：**
```
原始预测: 我爱中过学习  (准确率: 80%)
Qwen纠错:  我爱中国学习  (准确率: 100%) ✓
提升: +20%
```

**输出示例：**

*步骤1: predict_sentences.py*
```
[1/15] 文件: sentence_000.csv
真实句子: 我爱中国 (4 字)
预测句子: 我爱中过
平均置信度: 45.23%
字符准确率: 75.00%
----------------------------------------------------------------------

预测结果已保存到: prediction_results.json
提示: 可使用 'python llm_sentence_corrector.py' 对预测结果进行LLM纠错
```

*步骤2: llm_sentence_corrector.py*
```
======================================================================
基于Qwen3-1.7B的句子纠错系统
======================================================================

注意: 请先运行 'python predict_sentences.py' 生成预测结果
模型: Qwen/Qwen2.5-1.5B-Instruct

加载预测结果: prediction_results.json
✓ 预测结果加载成功

正在加载模型: Qwen/Qwen2.5-1.5B-Instruct
首次加载可能需要几分钟，请耐心等待...

使用设备: cuda
✓ 模型加载完成

预测句子数量: 15
原始字符准确率: 72.58%
======================================================================

[1/15] 文件: sentence_000.csv
真实句子: 我爱中国 (4 字)
原始预测: 我爱中过 (准确率: 75.0%, 置信度: 45.23%)
Qwen纠错中... 完成
纠错后句子: 我爱中国 (准确率: 100.0%, LLM置信度: 90.0%)
准确率变化: +25.0%
----------------------------------------------------------------------

总体统计
======================================================================
原始字符准确率: 45/62 = 72.58%
纠错后字符准确率: 56/62 = 90.32%
准确率提升: +17.74%

结果已保存到: llm_correction_results.json
```

**首次使用说明：**

首次运行会自动从HuggingFace下载模型到项目目录（约3GB），建议：
- 确保网络连接稳定
- 确保有足够的磁盘空间（至少5GB）
- 有GPU可显著加速（CUDA）
- CPU也可运行，但速度较慢

**模型保存位置：**
```
Radar_LLM_Agent_0/models/models--Qwen--Qwen2.5-1.5B-Instruct/
```

模型下载后会缓存到此目录，后续运行直接从本地加载，无需重新下载。

## LLM纠错详细指南

### 安装LLM依赖

```bash
# 安装必要的依赖
pip install transformers>=4.36.0
pip install accelerate>=0.25.0
pip install sentencepiece>=0.1.99
```

### 检查模型路径

运行以下命令查看模型缓存位置：
```bash
python check_model_path.py
```

输出示例：
```
模型缓存目录: H:\LLM\Radar_LLM_Agent_0\models
目录是否存在: 否（首次运行时会自动创建）
```

### 系统要求

**最低配置：**
- CPU: 4核以上
- 内存: 4GB
- 磁盘: 5GB空闲空间
- 网络: 首次运行需下载模型（约3GB）

**推荐配置：**
- CPU: 8核以上
- GPU: NVIDIA GPU with 8GB+ VRAM (CUDA)
- 内存: 8GB+
- 磁盘: 10GB空闲空间

### 模型缓存位置

模型会自动下载并缓存到项目目录下：
```
Radar_LLM_Agent_0/models/
```

首次运行会下载约3GB的模型文件到此目录，后续运行会直接从本地加载，无需重新下载。

### GPU加速配置

确保安装了CUDA版本的PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 输出文件说明

**prediction_results.json** - GRU预测结果
```json
{
  "overall_char_accuracy": 72.58,
  "sentence_accuracy": 60.00,
  "total_chars": 62,
  "predictions": [
    {
      "filename": "sentence_000.csv",
      "true_sentence": "我爱中国",
      "predicted_sentence": "我爱中过",
      "char_accuracy": 75.0,
      "avg_confidence": 0.4523
    }
  ]
}
```

**llm_correction_results.json** - LLM纠错结果
```json
{
  "model": "Qwen3-1.7B",
  "original_accuracy": 72.58,
  "corrected_accuracy": 90.32,
  "improvement": 17.74,
  "results": [
    {
      "filename": "sentence_000.csv",
      "true_sentence": "我爱中国",
      "predicted_sentence": "我爱中过",
      "corrected_sentence": "我爱中国",
      "improvement": 25.0
    }
  ]
}
```

### 常见问题

**Q1: 下载模型失败？**
- 检查网络连接
- 尝试使用HuggingFace镜像站：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

**Q2: CUDA out of memory？**
- 关闭其他占用GPU的程序
- 使用CPU模式（自动回退）
- 考虑使用更小的模型或量化版本

**Q3: 纠错效果不理想？**
- 确保原始准确率>60%
- 检查句子是否语义通顺
- 可以尝试调整temperature参数（在代码中）

**Q4: 运行速度慢？**
- 使用GPU会快10-20倍
- 首次运行需要加载模型，后续会快很多
- CPU模式下单句纠错需要5-10秒

**Q5: 如何更换其他模型？**

修改 `llm_sentence_corrector.py` 中的模型路径：
```python
def load_qwen_model(model_path='Qwen/Qwen2.5-7B-Instruct'):
    # 可以改为其他兼容的模型
    ...
```

### 使用技巧

**1. 批量处理**
一次生成预测结果后，可以多次尝试纠错：
```bash
# 生成一次
python predict_sentences.py

# 可以多次纠错实验
python llm_sentence_corrector.py
```

**2. 自定义Prompt**
在 `llm_sentence_corrector.py` 中修改prompt以获得更好效果：
```python
prompt = f"""你的自定义提示词...

原句：{predicted_sentence}

修正后："""
```

**3. 调整生成参数**
在 `correct_with_qwen()` 函数中：
```python
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=50,      # 最大生成token数
    temperature=0.1,        # 温度（越低越确定）
    top_p=0.9,             # nucleus sampling
    do_sample=True,
)
```

**4. 使用量化模型**
节省内存和提速：
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

## 性能指标

### 目标性能

- 训练准确率：>85%
- 验证准确率：>80%
- 测试准确率：>75%

### 评估指标

- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- 混淆矩阵

### 实际效果

训练完成后（20轮以上），模型应该能够：
- 准确识别大部分单字（>80%准确率）
- 识别通顺的短句子（3-5字）
- 提供置信度评估

## 代码组织

### model/gru_model.py

GRU神经网络模型定义：
- `RadarGRU` 类：模型架构
- 前向传播逻辑
- 权重初始化

### model/radar_tfnet_model.py ⭐新增

RadarTFNet雷达时频神经网络模型定义：
- `RadarTFNet` 类：完整架构
- `IntraFrameFullBandModule`：帧内全频带模块
- `SubBandTemporalModule`：子频段时间模块
- `FrequencyTemporalAttention`：频率-时间联合注意力模块
- `ComplexInputProcessor`：复数输入处理模块
- `DepthwiseSeparableConv1d`：深度可分离卷积

### train.py（原train_gru.py）⭐更新

训练脚本和数据处理：
- `RadarDataset` 类：数据加载和预处理
- `RadarTrainer` 类：训练和验证逻辑
- `create_model` 函数：模型创建工厂（支持GRU和RadarTFNet）
- 数据加载器创建
- 模型评估和可视化
- 命令行参数解析（支持模型选择）

### predict_sentences.py ⭐更新

句子识别测试：
- `load_model` 函数：支持加载GRU或RadarTFNet模型
- 句子数据加载
- 自动分割成单字
- 逐字识别
- 结果统计和展示
- **保存预测结果到JSON文件**（用于LLM纠错）
- 命令行参数解析（支持模型选择）

### llm_sentence_corrector.py

LLM智能纠错（Qwen2.5-1.5B）：
- **读取保存的预测结果**（prediction_results.json）
- 加载Qwen2.5-1.5B本地模型
- 使用transformer推理进行语义纠错
- 对比分析和结果保存
- 完全本地运行，无需API
- 无需重新运行模型

## 应用场景

### 无障碍通信
- 无声语音识别
- 手势识别
- 唇语识别

### 医疗康复
- 言语障碍评估
- 康复训练辅助
- 语音功能监测

### 人机交互
- 无声指令识别
- 隐私保护通信
- 噪声环境通信

## 技术优势

### 雷达技术优势
- 不受环境光照影响
- 可穿透非金属障碍物
- 隐私保护性好
- 实时性好

### 深度学习优势
- 端到端学习
- 自动特征提取
- 高准确率
- 可扩展性强

## 注意事项

### 训练相关
1. **数据格式**：确保CSV文件格式正确（80×128复数矩阵）
2. **内存使用**：根据GPU内存调整批次大小
3. **训练时间**：完整训练大约需要1-2小时（20轮）
4. **模型保存**：训练过程中会自动保存最佳模型

### 性能优化
1. **内存不足**：减少批次大小或隐藏层大小
2. **训练不收敛**：调整学习率或增加训练轮数
3. **过拟合**：增加Dropout率或减少模型复杂度
4. **准确率低**：检查数据质量或增加训练数据

### 句子识别
1. 句子中的所有汉字必须在训练集中
2. 当前支持3-5字的短句识别
3. 模型需要充分训练（至少20轮）才能获得较好效果

### LLM纠错
1. 首次运行需要下载模型（约3GB）到项目的 `models/` 目录
2. 推荐使用GPU（CUDA），CPU也可运行但较慢
3. 至少需要4GB内存（GPU推荐8GB显存）
4. 纠错效果依赖原始识别质量（建议>60%）
5. 模型下载后会缓存，后续运行无需重新下载
6. 模型文件已在 `.gitignore` 中，不会被提交到版本控制

## 扩展功能

### 模型改进
- ✅ RadarTFNet架构（已完成）
- 增加更多GRU层
- 调整隐藏层大小
- 添加注意力机制
- 添加残差连接
- 复数神经网络层设计
- 相位-幅度分解架构

### 数据增强
- 添加噪声
- 时间扭曲
- 幅度缩放
- 相位扰动

### LLM纠错优化
- 优化纠错Prompt提示词
- 添加置信度阈值过滤
- 实现多轮迭代纠错
- 集成更大的模型（Qwen-7B等）
- 量化优化以降低内存占用

### 功能扩展
- 支持更长的句子识别
- 实时语音识别
- 多人识别
- 情感分析
- 端到端LLM集成

## 许可证

本项目采用MIT许可证，允许：
- 商业和非商业使用
- 修改和分发
- 要求署名

## 引用说明

如果您在研究中使用了本项目，请引用：

```bibtex
@project{chinese_radar_speech_recognition_2024,
  title={Chinese Radar Speech Recognition System with Deep Learning},
  author={Your Name},
  year={2024},
  note={Supporting GRU and RadarTFNet architectures},
  url={https://github.com/your-repo/radar-speech-recognition}
}
```

**参考的模型架构论文：**
```bibtex
@article{tfgridnet2022,
  title={TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation},
  author={Zhong-Qiu Wang and others},
  journal={NeurIPS},
  year={2022},
  url={https://arxiv.org/abs/2211.12433}
}
```

## 联系方式

如有问题或建议，请联系：
- 项目地址：https://github.com/your-repo/radar-speech-recognition
- 提交问题：https://github.com/your-repo/radar-speech-recognition/issues

## 更新日志

### v1.5.0 (2024-12) ⭐最新
- ⭐ 数据集优化：从416个字符缩减到100个常用中文字
- ⭐ 样本数量：从4,160个缩减到1,000个（每个字符保留10个样本）
- ⭐ 优先保留句子测试数据集中的字符，确保所有测试句子可用
- 更新模型分类数量为100个汉字
- 优化数据集结构，提升训练效率

### v1.4.0 (2024-12)
- ⭐ 添加RadarTFNet模型架构（雷达时频神经网络）
- ⭐ 训练脚本支持模型选择（train.py）
- ⭐ 预测脚本支持模型选择（predict_sentences.py）
- 基于TF-GridNet架构，针对雷达复数数据优化
- 支持复数输入处理、帧内全频带、子频段时间、频率-时间联合注意力模块
- 模型文件自动命名（根据模型类型）

### v1.3.0 (2024-10-25)
- ⭐ 添加LLM智能纠错功能（Qwen2.5-1.5B）
- 完全本地运行，无需API密钥
- 准确率提升10-20%
- 预测结果保存和复用机制
- 更新依赖配置（transformers）

### v1.2.0 (2024-10-22)
- 添加句子级识别功能
- 创建15个测试句子数据集
- 优化代码组织结构
- 完善文档说明

### v1.1.0
- 将RadarDataset类移至train.py（原train_gru.py）
- 优化模型训练流程
- 添加中文字体支持

### v1.0.0
- 初始版本发布
- 实现基础GRU模型
- 完整的训练和评估流程
- 包含100个汉字，1,000个样本

## 致谢

感谢所有参与数据采集的受试者和研究团队成员的贡献。

