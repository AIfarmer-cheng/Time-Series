#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Qwen2.5-1.5B的句子纠错系统
独立运行，对雷达识别的句子进行语义修复

使用方法：
1. 先运行 python inference/predict_sentences.py 生成预测结果
2. 再运行本脚本进行LLM纠错

模型: Qwen2.5-1.5B (本地运行)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==================== 全局模型加载 ====================

# 全局变量，避免重复加载模型
_model = None
_tokenizer = None
_device = None


def load_qwen_model(model_path='Qwen/Qwen2.5-1.5B-Instruct'):
    """
    加载Qwen2.5-1.5B模型（仅加载一次）
    
    Args:
        model_path: 模型路径或HuggingFace模型ID
    
    Returns:
        model, tokenizer, device
    """
    global _model, _tokenizer, _device # 声明全局变量，只加载一次
    
    if _model is not None:
        return _model, _tokenizer, _device # 如果模型已加载，直接返回
    
    # 设置模型缓存目录为当前项目下的 models/ 文件夹
    cache_dir = Path(__file__).parent / 'models'
    cache_dir.mkdir(exist_ok=True)
    
    print(f"正在加载模型: {model_path}")
    print(f"模型缓存目录: {cache_dir.absolute()}")
    print("首次加载可能需要几分钟，请耐心等待...\n")
    
    # 确定设备
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {_device}")
    
    # 加载tokenizer，文本转token
    _tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=str(cache_dir),
        trust_remote_code=True
    )
    
    # 加载模型
    _model = AutoModelForCausalLM.from_pretrained( # 因果语言模型
        model_path,
        cache_dir=str(cache_dir),
        dtype=torch.float16 if _device == 'cuda' else torch.float32,
        device_map='auto' if _device == 'cuda' else None,
        trust_remote_code=True
    )
    
    if _device == 'cpu':
        _model = _model.to(_device)
    
    _model.eval()
    
    print("✓ 模型加载完成\n")
    
    return _model, _tokenizer, _device


# ==================== LLM 纠错模块 ====================

def correct_sentence_with_llm(predicted_sentence, true_sentence=None):
    """
    使用Qwen2.5-1.5B纠错句子
    
    Args:
        predicted_sentence: 模型预测的句子（可能有错误）
        true_sentence: 真实句子（可选，用于参考，但不传给模型）
    
    Returns:
        corrected_sentence: 纠错后的句子
        confidence: 纠错置信度
    """
    return correct_with_qwen(predicted_sentence)


def correct_with_qwen(predicted_sentence):
    """
    使用Qwen2.5-1.5B本地模型进行纠错
    """
    try:
        # 加载模型（首次调用时加载，后续复用）
        model, tokenizer, device = load_qwen_model()
        
        # 构建prompt提示文本
        prompt = f"""你是一个中文句子纠错助手。下面这句话是从雷达语音识别得到的，可能有个别汉字识别错误。
请基于语义和上下文，将其修正为通顺、合理的中文句子。

要求：
1. 只修正明显的错误汉字
2. 保持原句的长度和大致意思
3. 只输出修正后的句子，不要解释

原句：{predicted_sentence}

修正后："""
        
        # 使用chat模板
        messages = [
            {"role": "system", "content": "你是一个专业的中文句子纠错助手。"}, # 设定AI
            {"role": "user", "content": prompt} # 设定用户
        ]
        # 模型的聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Tokenize编码（文本->数字）
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        # 生成回答
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=50,
                temperature=0.1, # 温度，控制生成文本的随机性
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        # 只保留生成的token
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # 解码（数字->文本）
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 清理输出
        corrected = response.strip()
        # 移除可能的引号和多余标点
        corrected = corrected.replace('"', '').replace('"', '').replace('"', '').strip()
        # 只取第一行（避免模型输出解释）
        corrected = corrected.split('\n')[0].strip()
        
        # 如果输出为空或太长，返回原句
        if not corrected or len(corrected) > len(predicted_sentence) * 2:
            return predicted_sentence, 0.5
        
        return corrected, 0.9
        
    except Exception as e:
        print(f"Qwen纠错失败: {e}")
        print(f"返回原句: {predicted_sentence}")
        return predicted_sentence, 0.0


# ==================== 主要功能 ====================

def load_prediction_results(prediction_file=None):
    """
    加载预测结果文件
    
    Args:
        prediction_file: 预测结果JSON文件路径
    
    Returns:
        预测结果数据字典
    """
    if prediction_file is None:
        prediction_file = PROJECT_ROOT / 'results' / 'prediction_results.json'
    
    if not Path(prediction_file).exists():
        raise FileNotFoundError(
            f"预测结果文件 {prediction_file} 不存在！\n"
            f"请先运行: python inference/predict_sentences.py"
        )
    
    with open(prediction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def test_with_llm_correction(prediction_file='prediction_results.json'):
    """
    使用Qwen2.5-1.5B纠错测试句子识别
    
    Args:
        prediction_file: 预测结果JSON文件路径
    """
    print("="*70)
    print("基于Qwen2.5-1.5B的句子纠错系统")
    print("="*70 + "\n")
    
    # 加载预测结果
    try:
        print(f"加载预测结果: {prediction_file}")
        prediction_data = load_prediction_results(prediction_file)
        print("✓ 预测结果加载成功\n")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    print(f"预测句子数量: {prediction_data['total_sentences']}")
    print(f"原始字符准确率: {prediction_data['overall_char_accuracy']:.2f}%\n")
    print("="*70)
    
    # 统计结果
    results = []
    original_correct = prediction_data['total_char_correct']
    corrected_correct = 0
    total_chars = prediction_data['total_chars']
    
    # 逐个纠错句子
    for idx, pred in enumerate(prediction_data['predictions']):
        filename = pred['filename']
        true_sentence = pred['true_sentence']
        predicted_sentence = pred['predicted_sentence']
        avg_confidence = pred['avg_confidence']
        original_accuracy = pred['char_accuracy']
        
        print(f"\n[{idx+1}/{prediction_data['total_sentences']}] 文件: {filename}")
        print(f"真实句子: {true_sentence} ({len(true_sentence)} 字)")
        print(f"原始预测: {predicted_sentence} (准确率: {original_accuracy:.1f}%, 置信度: {avg_confidence*100:.2f}%)")
        
        # LLM纠错
        print(f"Qwen纠错中...", end='', flush=True)
        corrected_sentence, llm_confidence = correct_sentence_with_llm(
            predicted_sentence, true_sentence
        )
        print(f" 完成")
        
        # 纠错后准确率
        if len(corrected_sentence) == len(true_sentence):
            corrected_accuracy = sum(1 for c, t in zip(corrected_sentence, true_sentence) if c == t) / len(true_sentence) * 100
            corrected_correct += sum(1 for c, t in zip(corrected_sentence, true_sentence) if c == t)
        else:
            corrected_accuracy = 0
        
        improvement = corrected_accuracy - original_accuracy
        
        print(f"纠错后句子: {corrected_sentence} (准确率: {corrected_accuracy:.1f}%, LLM置信度: {llm_confidence*100:.1f}%)")
        print(f"准确率变化: {improvement:+.1f}%")
        
        # 保存结果
        results.append({
            'filename': filename,
            'true_sentence': true_sentence,
            'predicted_sentence': predicted_sentence,
            'corrected_sentence': corrected_sentence,
            'original_accuracy': original_accuracy,
            'corrected_accuracy': corrected_accuracy,
            'improvement': improvement,
            'avg_confidence': avg_confidence,
            'llm_confidence': llm_confidence
        })
        
        print("-" * 70)
    
    # 总体统计
    print("\n" + "="*70)
    print("总体统计")
    print("="*70)
    
    original_char_acc = original_correct / total_chars * 100
    corrected_char_acc = corrected_correct / total_chars * 100
    improvement = corrected_char_acc - original_char_acc
    
    print(f"\n原始字符准确率: {original_correct}/{total_chars} = {original_char_acc:.2f}%")
    print(f"纠错后字符准确率: {corrected_correct}/{total_chars} = {corrected_char_acc:.2f}%")
    print(f"准确率提升: {improvement:+.2f}%")
    
    # 详细结果
    print("\n详细对比:")
    print("-" * 70)
    print(f"{'真实句子':<15} {'原始预测':<15} {'LLM纠错':<15} {'提升':<8}")
    print("-" * 70)
    
    for r in results:
        improvement_str = f"{r['improvement']:+.1f}%"
        print(f"{r['true_sentence']:<15} {r['predicted_sentence']:<15} {r['corrected_sentence']:<15} {improvement_str:<8}")
    
    print("-" * 70)
    
    # 保存结果到文件
    output_file = 'llm_correction_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'Qwen2.5-1.5B',
            'original_accuracy': original_char_acc,
            'corrected_accuracy': corrected_char_acc,
            'improvement': improvement,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


def main():
    """主函数"""
    print("="*70)
    print("基于Qwen2.5-1.5B的句子纠错系统")
    print("="*70)
    print()
    print("注意: 请先运行 'python inference/predict_sentences.py' 生成预测结果")
    print("模型: Qwen/Qwen2.5-1.5B-Instruct")
    print()
    
    test_with_llm_correction(prediction_file=None)


if __name__ == "__main__":
    main()

