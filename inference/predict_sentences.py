#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别句子测试数据集
支持GRU、RadarTFNet、ResNet、VGG、ConvNeXt、TFConv、EfficientNet-V2和Vision Transformer八种模型
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from model.gru_model import RadarGRU
from model.radar_tfnet_model import RadarTFNet
from model.resnet import resnet34
from model.vggnet import vgg11
from model.convnext import convnext_tiny
from model.convnextTF import tfconv_tiny
from model.efficientnet_v2 import efficientnetv2_s
from model.vision_transformer import vit_base_patch8_80x128
from data_utils import RadarDataset


def load_model(model_type, model_path, num_classes=416, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载训练好的模型
    
    Args:
        model_type: 'gru', 'radar_tfnet', 'resnet', 'vgg', 'convnext', 'tfconv', 'efficientnet' 或 'vit'
        model_path: 模型文件路径
        num_classes: 分类数量
        device: 设备
    """
    if model_type.lower() == 'gru':
        model = RadarGRU(
            input_size=256,  # 128 bins × 2 channels (实部+虚部)
            hidden_size=64,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        )
    elif model_type.lower() == 'radar_tfnet':
        model = RadarTFNet(
            num_classes=num_classes,
            input_channels=2,
            freq_bins=128,
            time_frames=80,
            base_channels=64,
            num_subbands=4,
            num_layers=2
        )
    elif model_type.lower() == 'resnet':
        model = resnet34(num_classes=num_classes, include_top=True, input_channels=2)
    elif model_type.lower() == 'vgg':
        model = vgg11(num_classes=num_classes, input_channels=2, use_adaptive_pool=True)
    elif model_type.lower() == 'convnext':
        model = convnext_tiny(num_classes=num_classes, in_chans=2)
    elif model_type.lower() == 'tfconv':
        model = tfconv_tiny(num_classes=num_classes, in_chans=2)
    elif model_type.lower() == 'efficientnet':
        model = efficientnetv2_s(num_classes=num_classes, in_chans=2)
    elif model_type.lower() == 'vit':
        model = vit_base_patch8_80x128(num_classes=num_classes, in_c=2)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。请选择 'gru', 'radar_tfnet', 'resnet', 'vgg', 'convnext', 'tfconv', 'efficientnet' 或 'vit'")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


def load_sentence_radar_data(filepath):
    """加载句子雷达数据"""
    df = pd.read_csv(filepath, header=None)
    return df


def split_sentence_into_chars(sentence_data, char_height=80, char_width=128):
    """
    将句子雷达数据分割成单个汉字
    
    Args:
        sentence_data: 整个句子的雷达数据
        char_height: 单个汉字的高度（行数）
        char_width: 单个汉字的宽度（列数）
    
    Returns:
        汉字雷达数据列表
    """
    total_rows = len(sentence_data)
    num_chars = total_rows // char_height
    
    char_data_list = []
    
    for i in range(num_chars):
        start_row = i * char_height
        end_row = start_row + char_height
        char_data = sentence_data.iloc[start_row:end_row]
        char_data_list.append(char_data)
    
    return char_data_list


def process_char_data(char_df):
    """
    将单个汉字的DataFrame转换为模型输入格式
    
    Args:
        char_df: 80x128的复数雷达数据
    
    Returns:
        numpy: shape=(2, 80, 128)
    """
    height, width = char_df.shape
    
    radar_data = np.zeros((height, width), dtype=complex)
    for i in range(height):
        for j in range(width):
            complex_str = char_df.iloc[i, j]
            radar_data[i, j] = complex(complex_str) # 将字符串转换为复数
    real_part = radar_data.real
    imag_part = radar_data.imag
    data = np.stack([real_part, imag_part], axis=0)
    
    return data


def predict_char(model, radar_data, device):
    """预测单个汉字"""
    with torch.no_grad():
        data = torch.from_numpy(radar_data).float().unsqueeze(0).to(device) # 维度扩展，[1, 2, 80, 128]
        output = model(data) # [1, 416]
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item()


def predict_sentence(model, dataset, sentence_filepath, device):
    """
    识别一个句子
    
    Args:
        model: 训练好的模型
        dataset: 数据集对象（用于获取字符映射）
        sentence_filepath: 句子CSV文件路径
        device: 设备
    
    Returns:
        识别结果字典
    """
    # 加载句子雷达数据
    sentence_data = load_sentence_radar_data(sentence_filepath) # pd.read_csv()
    
    # 分割成单个汉字
    char_data_list = split_sentence_into_chars(sentence_data)
    
    # 逐字识别
    predicted_chars = []
    confidences = []
    
    for i, char_df in enumerate(char_data_list):
        # 转换为模型输入格式
        radar_data = process_char_data(char_df)
        
        # 预测
        predicted_label, confidence = predict_char(model, radar_data, device) # confidence置信度
        predicted_char = dataset.get_char_name(predicted_label)
        
        predicted_chars.append(predicted_char)
        confidences.append(confidence)
    
    return {
        'predicted_chars': predicted_chars,
        'predicted_sentence': ''.join(predicted_chars),
        'confidences': confidences,
        'avg_confidence': np.mean(confidences),
        'num_chars': len(predicted_chars)
    }


def evaluate_predictions(predictions, true_sentence):
    """评估预测结果"""
    predicted = predictions['predicted_sentence']
    
    # 计算字符级准确率
    if len(predicted) != len(true_sentence):
        char_accuracy = 0.0
        correct_positions = []
    else:
        correct_positions = [i for i in range(len(true_sentence)) if predicted[i] == true_sentence[i]]
        char_accuracy = len(correct_positions) / len(true_sentence) * 100
    
    # 句子级准确率
    sentence_correct = (predicted == true_sentence)
    
    return {
        'char_accuracy': char_accuracy,
        'sentence_correct': sentence_correct,
        'correct_positions': correct_positions
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='句子识别测试')
    parser.add_argument(
        '--model',
        type=str,
        default='tfconv',
        choices=['gru', 'radar_tfnet', 'resnet', 'vgg', 'convnext', 'tfconv', 'efficientnet', 'vit'],
        help='选择模型类型: gru, radar_tfnet, resnet, vgg, convnext, tfconv, efficientnet 或 vit (默认: radar_tfnet)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("句子识别测试")
    print("="*70)
    print(f"模型类型: {args.model.upper()}")
    print("="*70 + "\n")
    
    # 根据模型类型确定模型文件路径
    model_name = args.model.lower()
    if model_name == 'radar_tfnet':
        model_path = PROJECT_ROOT / "model" / "best_radar_tfnet_model.pth"
    elif model_name == 'resnet':
        model_path = PROJECT_ROOT / "model" / "best_resnet_model.pth"
    elif model_name == 'vgg':
        model_path = PROJECT_ROOT / "model" / "best_vgg_model.pth"
    elif model_name == 'convnext':
        model_path = PROJECT_ROOT / "model" / "best_convnext_model.pth"
    elif model_name == 'tfconv':
        model_path = PROJECT_ROOT / "model" / "best_tfconv_model.pth"
    elif model_name == 'efficientnet':
        model_path = PROJECT_ROOT / "model" / "best_efficientnet_model.pth"
    elif model_name == 'vit':
        model_path = PROJECT_ROOT / "model" / "best_vit_model.pth"
    else:
        model_path = PROJECT_ROOT / "model" / "best_gru_model.pth"
    
    if not model_path.exists():
        print(f"错误: 模型文件 {model_path} 不存在！")
        print(f"请先运行 'python training/train.py --model {args.model}' 训练模型")
        return
    
    # 检查句子数据集
    sentence_dir = PROJECT_ROOT / "sentence_test_dataset"
    if not sentence_dir.exists():
        print(f"错误: 句子数据集目录 {sentence_dir} 不存在！")
        return
    
    labels_file = sentence_dir / "sentence_labels.csv"
    if not labels_file.exists():
        print(f"错误: 标签文件 {labels_file} 不存在！")
        return
    
    # 加载数据集
    print("加载训练数据集（用于字符映射）...")
    dataset = RadarDataset(PROJECT_ROOT / 'chinese_radar_dataset', 'labels.csv', 'characters.csv')
    
    # 加载模型
    print(f"加载{args.model.upper()}模型...")
    print(f"模型路径: {model_path}")
    model, device = load_model(args.model, model_path, num_classes=len(dataset.chars_df))
    print(f"使用设备: {device}\n")
    
    # 加载句子标签
    sentence_labels = pd.read_csv(labels_file)
    
    print(f"找到 {len(sentence_labels)} 个测试句子\n")
    print("="*70)
    
    # 统计结果
    all_results = []
    total_char_correct = 0
    total_chars = 0
    sentence_correct_count = 0
    
    # 逐个识别句子
    for idx, row in sentence_labels.iterrows():
        filename = row['filename']
        true_sentence = row['sentence']
        filepath = sentence_dir / 'data' / filename
        
        print(f"\n[{idx+1}/{len(sentence_labels)}] 文件: {filename}")
        print(f"真实句子: {true_sentence} ({len(true_sentence)} 字)")
        
        # 预测
        predictions = predict_sentence(model, dataset, filepath, device)
        
        # 评估
        eval_results = evaluate_predictions(predictions, true_sentence)
        
        # 显示结果
        print(f"预测句子: {predictions['predicted_sentence']}")
        print(f"平均置信度: {predictions['avg_confidence']*100:.2f}%")
        print(f"字符准确率: {eval_results['char_accuracy']:.2f}%")
        
        # 逐字对比
        print("\n逐字对比:")
        print("-" * 70)
        print(f"{'位置':<6} {'真实':<8} {'预测':<8} {'结果':<8} {'置信度':<12}")
        print("-" * 70)
        
        for i in range(len(true_sentence)):
            true_char = true_sentence[i]
            pred_char = predictions['predicted_chars'][i] if i < len(predictions['predicted_chars']) else 'N/A'
            confidence = predictions['confidences'][i] if i < len(predictions['confidences']) else 0
            is_correct = (true_char == pred_char)
            status = "OK" if is_correct else "FAIL"
            
            print(f"{i+1:<6} {true_char:<8} {pred_char:<8} {status:<8} {confidence*100:>6.2f}%")
        
        print("-" * 70)
        
        # 更新统计
        if len(predictions['predicted_sentence']) == len(true_sentence):
            total_char_correct += len(eval_results['correct_positions'])
        total_chars += len(true_sentence)
        
        if eval_results['sentence_correct']:
            sentence_correct_count += 1
        
        all_results.append({
            'filename': filename,
            'true_sentence': true_sentence,
            'predicted_sentence': predictions['predicted_sentence'],
            'predicted_chars': predictions['predicted_chars'],
            'confidences': predictions['confidences'],
            'char_accuracy': eval_results['char_accuracy'],
            'sentence_correct': eval_results['sentence_correct'],
            'avg_confidence': predictions['avg_confidence']
        })
    
    # 总体统计
    print("\n" + "="*70)
    print("总体统计")
    print("="*70)
    
    overall_char_accuracy = total_char_correct / total_chars * 100 if total_chars > 0 else 0
    sentence_accuracy = sentence_correct_count / len(sentence_labels) * 100 if len(sentence_labels) > 0 else 0
    
    print(f"\n字符级准确率: {total_char_correct}/{total_chars} = {overall_char_accuracy:.2f}%")
    print(f"句子级准确率: {sentence_correct_count}/{len(sentence_labels)} = {sentence_accuracy:.2f}%")
    
    print("\n详细结果:")
    print("-" * 70)
    for result in all_results:
        status = "OK" if result['sentence_correct'] else "FAIL"
        print(f"[{status}] {result['true_sentence']:8s} -> {result['predicted_sentence']:8s} "
              f"(准确率: {result['char_accuracy']:5.1f}%, 置信度: {result['avg_confidence']*100:5.2f}%)")
    print("-" * 70)
    
    # 保存预测结果
    import json
    output_file = PROJECT_ROOT / 'results' / 'prediction_results.json'
    save_data = {
        'overall_char_accuracy': overall_char_accuracy,
        'sentence_accuracy': sentence_accuracy,
        'total_chars': total_chars,
        'total_char_correct': total_char_correct,
        'sentence_correct_count': sentence_correct_count,
        'total_sentences': len(sentence_labels),
        'predictions': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n预测结果已保存到: {output_file}")
    print("提示: 可使用 'python llm_sentence_corrector.py' 对预测结果进行LLM纠错")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()

