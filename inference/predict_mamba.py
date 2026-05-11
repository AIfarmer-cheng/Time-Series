#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba句子识别测试
支持语义增强Mamba和普通Mamba两种模式
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import argparse
import json

from model.semantic_mamba import SemanticMamba
from data_utils import RadarDataset


def load_mamba_model(model_path, num_classes, use_semantic=True,
                     d_model=192, num_blocks=4,
                     device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载Mamba模型"""
    model = SemanticMamba(
        num_classes=num_classes,
        d_model=d_model,
        num_mamba_blocks=num_blocks,
        use_semantic=use_semantic,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    # 兼容旧模型权重（没有class_prototypes的情况）
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device


def load_sentence_radar_data(filepath):
    """加载句子雷达数据"""
    df = pd.read_csv(filepath, header=None)
    return df


def split_sentence_into_chars(sentence_data, char_height=80, char_width=128):
    """将句子雷达数据分割成单个汉字"""
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
    """将单个汉字的DataFrame转换为模型输入格式 (2, 80, 128)"""
    height, width = char_df.shape
    radar_data = np.zeros((height, width), dtype=complex)
    for i in range(height):
        for j in range(width):
            complex_str = char_df.iloc[i, j]
            radar_data[i, j] = complex(complex_str)
    real_part = radar_data.real
    imag_part = radar_data.imag
    data = np.stack([real_part, imag_part], axis=0)
    return data


def predict_char(model, radar_data, device):
    """预测单个汉字"""
    with torch.no_grad():
        data = torch.from_numpy(radar_data).float().unsqueeze(0).to(device)
        output = model(data)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item()


def predict_sentence(model, dataset, sentence_filepath, device):
    """识别一个句子"""
    sentence_data = load_sentence_radar_data(sentence_filepath)
    char_data_list = split_sentence_into_chars(sentence_data)

    predicted_chars = []
    confidences = []

    for char_df in char_data_list:
        radar_data = process_char_data(char_df)
        predicted_label, confidence = predict_char(model, radar_data, device)
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

    if len(predicted) != len(true_sentence):
        char_accuracy = 0.0
        correct_positions = []
    else:
        correct_positions = [i for i in range(len(true_sentence))
                             if predicted[i] == true_sentence[i]]
        char_accuracy = len(correct_positions) / len(true_sentence) * 100

    sentence_correct = (predicted == true_sentence)

    return {
        'char_accuracy': char_accuracy,
        'sentence_correct': sentence_correct,
        'correct_positions': correct_positions
    }


def main():
    parser = argparse.ArgumentParser(description='Mamba句子识别测试')
    parser.add_argument('--no_semantic', action='store_true',
                        help='关闭语义增强，使用纯雷达Mamba推理')
    parser.add_argument('--d_model', type=int, default=192,
                        help='模型维度 (默认192)')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Mamba块数 (默认4)')
    args = parser.parse_args()

    print("=" * 70)
    print("Mamba句子识别测试")
    print("=" * 70)
    model_tag = 'semantic' if not args.no_semantic else 'no_semantic'
    print(f"语义增强: {'关闭' if args.no_semantic else '开启'}")
    print("=" * 70 + "\n")

    # 模型路径
    model_path = PROJECT_ROOT / "model" / f"best_{model_tag}_mamba.pth"
    if not model_path.exists():
        print(f"错误: 模型文件 {model_path} 不存在！")
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

    # 加载数据集（用于字符映射）
    print("加载训练数据集（用于字符映射）...")
    dataset = RadarDataset(PROJECT_ROOT / 'chinese_radar_dataset',
                           'labels.csv', 'characters.csv')

    # 加载模型
    print(f"加载Mamba模型...")
    print(f"模型路径: {model_path}")
    model, device = load_mamba_model(
        model_path,
        num_classes=dataset.num_classes,
        use_semantic=not args.no_semantic,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
    )
    print(f"使用设备: {device}\n")

    # 加载句子标签
    sentence_labels = pd.read_csv(labels_file)

    print(f"找到 {len(sentence_labels)} 个测试句子\n")
    print("=" * 70)

    all_results = []
    total_char_correct = 0
    total_chars = 0
    sentence_correct_count = 0

    for idx, row in sentence_labels.iterrows():
        filename = row['filename']
        true_sentence = row['sentence']
        filepath = sentence_dir / 'data' / filename

        print(f"\n[{idx + 1}/{len(sentence_labels)}] 文件: {filename}")
        print(f"真实句子: {true_sentence} ({len(true_sentence)} 字)")

        predictions = predict_sentence(model, dataset, filepath, device)
        eval_results = evaluate_predictions(predictions, true_sentence)

        print(f"预测句子: {predictions['predicted_sentence']}")
        print(f"平均置信度: {predictions['avg_confidence'] * 100:.2f}%")
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
            print(f"{i + 1:<6} {true_char:<8} {pred_char:<8} {status:<8} {confidence * 100:>6.2f}%")

        print("-" * 70)

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
    print("\n" + "=" * 70)
    print("总体统计")
    print("=" * 70)

    overall_char_accuracy = total_char_correct / total_chars * 100 if total_chars > 0 else 0
    sentence_accuracy = sentence_correct_count / len(sentence_labels) * 100 if len(sentence_labels) > 0 else 0

    print(f"\n字符级准确率: {total_char_correct}/{total_chars} = {overall_char_accuracy:.2f}%")
    print(f"句子级准确率: {sentence_correct_count}/{len(sentence_labels)} = {sentence_accuracy:.2f}%")

    print("\n详细结果:")
    print("-" * 70)
    for result in all_results:
        status = "OK" if result['sentence_correct'] else "FAIL"
        print(f"[{status}] {result['true_sentence']:8s} -> {result['predicted_sentence']:8s} "
              f"(准确率: {result['char_accuracy']:5.1f}%, 置信度: {result['avg_confidence'] * 100:5.2f}%)")
    print("-" * 70)

    # 保存结果
    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f'prediction_{model_tag}_mamba_results.json'
    save_data = {
        'model': f'mamba_{model_tag}',
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

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
