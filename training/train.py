#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模型的脚本
支持GRU、RadarTFNet、ResNet、VGG、ConvNeXt、TFConv、EfficientNet-V2和Vision Transformer八种模型
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from model.gru_model import RadarGRU
# from model.radar_tfnet_model import RadarTFNet
from model.resnet import resnet34
from model.vggnet import vgg11
from model.convnext import convnext_tiny
from model.convnextTF import tfconv_tiny
from model.efficientnet_v2 import efficientnetv2_s
from model.vision_transformer import vit_base_patch8_80x128
from data_utils import RadarDataset


class RadarTrainer:
    """雷达模型训练器"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
       
        self.model = model.to(device)
        self.device = device
        self.train_losses = []  # 训练损失列表
        self.val_losses = []
        self.train_accuracies = []  # 训练准确率列表
        self.val_accuracies = []

    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        batch_group_start_time = time.time()  # 记录每10个batch组的开始时间

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()  # 梯度清0
            output = self.model(data)  # 前向传播，[batch_size, num_classes]
            loss = criterion(output, target)  # 计算损失
            # 添加门正则项
            gate_loss = self.model.compute_gate_regularization(lambda_reg=0.01)
            loss += gate_loss
            loss.backward()  # 反向传播
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加损失，item将张量转换为数值
            _, predicted = torch.max(output.data, 1)  # 返回最大值和索引
            total += target.size(0)  # 样本数量
            correct += (predicted == target).sum().item()  # sum布尔值求和

            if batch_idx % 10 == 0:  # 每10个批次打印一次损失，416*8=3328个训练csv，3328/8=416个批次一轮训练
                # 计算这10个batch的总耗时
                batch_group_time = time.time() - batch_group_start_time
                minutes = int(batch_group_time // 60)
                seconds = int(batch_group_time % 60)
                if minutes > 0:
                    time_str = f"{minutes}分钟{seconds}秒"
                else:
                    time_str = f"{seconds}秒"
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Time: {time_str}')
                batch_group_start_time = time.time()  # 重置10个batch组的开始时间

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate_epoch(self, val_loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs=30, learning_rate=0.001):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        # 调参
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 带动量的Adam优化器
        # optimizer = optim.SGD(
        #     self.model.parameters(),
        #     lr=learning_rate,
        #     momentum=0.9,       # 动量，0.9表示90%的当前梯度，10%的过去梯度
        #     weight_decay=1e-4   # 权重衰减，防止过拟合
        # )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # 学习率调度器
        # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs) # 余弦退火学习率调整器，自动调整lr

        print(f"开始训练，设备: {self.device}")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")

        best_val_acc = 0

        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)

            # 学习率调度
            scheduler.step(val_loss)
            # scheduler.step()

            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 根据模型类型保存不同的文件名
                model_name = getattr(self.model, 'model_name', 'model')
                torch.save(self.model.state_dict(), PROJECT_ROOT / 'model' / f'best_{model_name}_model.pth')

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
            print(f'  当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)

        print(f'训练完成！最佳验证准确率: {best_val_acc:.2f}%')

    def plot_training_history(self, model_name='model'):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.val_losses, label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()  # 显示图例
        ax1.grid(True)  # 显示网格

        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率')
        ax2.plot(self.val_accuracies, label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('训练和验证准确率')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()  # 调整子图之间的间距
        plt.savefig(PROJECT_ROOT / 'model' / f'training_history_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_data_loaders(data_dir, batch_size=8, train_ratio=0.8, num_workers=4):  # 一个批次处理8个csv
    """
    创建训练和验证数据加载器

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        num_workers: 工作进程数

    Returns:
        (train_loader, val_loader, dataset): 训练和验证数据加载器及数据集
    """
    # 创建数据集
    dataset = RadarDataset(data_dir, 'labels.csv', 'characters.csv')

    # 划分训练和验证集
    total_size = len(dataset)  # 调用类的__len__方法
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )  # 采用随即划分，打乱

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 训练速度
        pin_memory=True  # 训练速度
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, dataset


def evaluate_model(model, test_loader, dataset, device='cuda' if torch.cuda.is_available() else 'cpu', model_name='model'):
    """评估模型"""
    model.eval()
    model = model.to(device)
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())  # 张量从GPU转移到CPU，转换为numpy数组，添加到列表中
            all_targets.extend(target.cpu().numpy())

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 获取实际出现的唯一类别
    unique_labels = sorted(set(all_targets) | set(all_predictions))  # set去重，sorted排序
    print(f'\n验证集中出现的类别数: {len(unique_labels)}/{len(dataset.chars_df)}')

    # 计算准确率
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    print(f'\n测试准确率: {accuracy:.2f}%')

    # 计算精确率、召回率、F1分数（宏平均和加权平均）
    precision_macro = precision_score(all_targets, all_predictions, labels=unique_labels, average='macro', zero_division=0) * 100
    precision_weighted = precision_score(all_targets, all_predictions, labels=unique_labels, average='weighted', zero_division=0) * 100
    
    recall_macro = recall_score(all_targets, all_predictions, labels=unique_labels, average='macro', zero_division=0) * 100
    recall_weighted = recall_score(all_targets, all_predictions, labels=unique_labels, average='weighted', zero_division=0) * 100
    
    f1_macro = f1_score(all_targets, all_predictions, labels=unique_labels, average='macro', zero_division=0) * 100
    f1_weighted = f1_score(all_targets, all_predictions, labels=unique_labels, average='weighted', zero_division=0) * 100

    print(f'\n精确率 (宏平均): {precision_macro:.2f}%')
    print(f'精确率 (加权平均): {precision_weighted:.2f}%')
    print(f'召回率 (宏平均): {recall_macro:.2f}%')
    print(f'召回率 (加权平均): {recall_weighted:.2f}%')
    print(f'F1分数 (宏平均): {f1_macro:.2f}%')
    print(f'F1分数 (加权平均): {f1_weighted:.2f}%')

    # 分类报告（只针对出现的类别）
    char_names = [dataset.get_char_name(i) for i in unique_labels]
    report = classification_report(
        all_targets,
        all_predictions,
        labels=unique_labels,
        target_names=char_names,
        zero_division=0,
        output_dict=True  # 返回字典格式以便提取加权平均
    )
    
    # 打印分类报告（文本格式）
    report_text = classification_report(
        all_targets,
        all_predictions,
        labels=unique_labels,
        target_names=char_names,
        zero_division=0
    )
    print('\n分类报告:')
    report_lines = report_text.split('\n')
    print('\n'.join(report_lines[:50]))
    if len(report_lines) > 50:
        print(f'... (省略 {len(report_lines) - 50} 行)')

    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions, labels=unique_labels)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(PROJECT_ROOT / 'model' / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存

    # 准备保存的结果
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report_text,
        'confusion_matrix': cm.tolist(),
        'unique_labels_count': len(unique_labels),
        'total_labels_count': len(dataset.chars_df)
    }

    # 保存结果到文件
    result_filename = PROJECT_ROOT / 'results' / f'{model_name}_train_result.txt'
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{model_name.upper()} 模型训练评估结果\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【总体评估指标】\n")
        f.write("-" * 70 + "\n")
        f.write(f"准确率 (Accuracy): {accuracy:.2f}%\n")
        f.write(f"精确率 (Precision - 宏平均): {precision_macro:.2f}%\n")
        f.write(f"精确率 (Precision - 加权平均): {precision_weighted:.2f}%\n")
        f.write(f"召回率 (Recall - 宏平均): {recall_macro:.2f}%\n")
        f.write(f"召回率 (Recall - 加权平均): {recall_weighted:.2f}%\n")
        f.write(f"F1分数 (F1-Score - 宏平均): {f1_macro:.2f}%\n")
        f.write(f"F1分数 (F1-Score - 加权平均): {f1_weighted:.2f}%\n\n")
        
        f.write(f"验证集中出现的类别数: {len(unique_labels)}/{len(dataset.chars_df)}\n\n")
        
        f.write("【分类报告】\n")
        f.write("-" * 70 + "\n")
        f.write(report_text)
        f.write("\n\n")
        
        f.write("【混淆矩阵】\n")
        f.write("-" * 70 + "\n")
        f.write(f"混淆矩阵形状: {cm.shape}\n")
        f.write("(行: 真实标签, 列: 预测标签)\n\n")
        
        # 保存混淆矩阵的数值（前20x20用于显示）
        if cm.shape[0] <= 20:
            f.write("完整混淆矩阵:\n")
            for i, row in enumerate(cm):
                f.write(f"{char_names[i] if i < len(char_names) else f'Class_{unique_labels[i]}'}: {row}\n")
        else:
            f.write("混淆矩阵过大，仅显示前20x20:\n")
            for i in range(min(20, cm.shape[0])):
                row_str = ' '.join([f'{val:4d}' for val in cm[i][:20]])
                f.write(f"{char_names[i] if i < len(char_names) else f'Class_{unique_labels[i]}'}: {row_str}...\n")
            f.write("(完整矩阵已保存在图像文件中)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"结果保存时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")

    print(f'\n评估结果已保存到: {result_filename}')

    return results


def create_model(model_type, num_classes):
    """
    创建模型

    Args:
        model_type: 'gru', 'radar_tfnet', 'resnet', 'vgg', 'convnext', 'tfconv', 'efficientnet' 或 'vit'
        num_classes: 分类数量

    Returns:
        模型实例
    """
    if model_type.lower() == 'gru':
        model = RadarGRU(
            input_size=256,  # 128 bins × 2 channels (实部+虚部)
            hidden_size=64,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        )
        model.model_name = 'gru'
    elif model_type.lower() == 'radar_tfnet':
        
        model.model_name = 'radar_tfnet'
    elif model_type.lower() == 'resnet': # 夯
        model = resnet34(num_classes=num_classes, include_top=True, input_channels=2)
        model.model_name = 'resnet'
    elif model_type.lower() == 'vgg':
        model = vgg11(num_classes=num_classes, input_channels=2, use_adaptive_pool=True)
        model.model_name = 'vgg'
    elif model_type.lower() == 'convnext': # 夯
        model = convnext_tiny(num_classes=num_classes, in_chans=2)
        model.model_name = 'convnext'
    elif model_type.lower() == 'tfconv':
        model = tfconv_tiny(num_classes=num_classes, in_chans=2)
        model.model_name = 'tfconv'
    elif model_type.lower() == 'efficientnet':
        model = efficientnetv2_s(num_classes=num_classes, in_chans=2)
        model.model_name = 'efficientnet'
    elif model_type.lower() == 'vit':
        model = vit_base_patch8_80x128(num_classes=num_classes, in_c=2)
        model.model_name = 'vit'
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。请选择 'gru', 'radar_tfnet', 'resnet', 'vgg', 'convnext', 'tfconv', 'efficientnet' 或 'vit'")

    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练雷达语音识别模型')
    parser.add_argument(
        '--model',
        type=str,
        default='tfconv',
        choices=['gru', 'radar_tfnet', 'resnet', 'vgg', 'convnext', 'tfconv', 'efficientnet', 'vit'],
        help='选择模型类型: gru, radar_tfnet, resnet, vgg, convnext, tfconv, efficientnet 或 vit (默认: tfconv)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='训练轮数 (默认: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='批次大小 (默认: 8)'
    )  # 一次吃进去多少才更新参数
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='学习率 (默认: 0.001)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("中文雷达语音识别模型训练")
    print("=" * 70)
    print(f"模型类型: {args.model.upper()}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print("=" * 70 + "\n")

    # 检查数据集
    data_dir = PROJECT_ROOT / "chinese_radar_dataset"
    if not data_dir.exists():
        print(f"数据集目录 {data_dir} 不存在！")
        return

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, dataset = create_data_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        train_ratio=0.8,
        num_workers=4
    )

    num_classes = len(dataset.chars_df)
    print(f"汉字数量: {num_classes}")
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}\n")

    # 创建模型
    print(f"创建{args.model.upper()}模型...")
    model = create_model(args.model, num_classes)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {params:,}")
    print()

    # 创建训练器
    trainer = RadarTrainer(model)

    # 训练模型
    print("开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )

    # 绘制训练历史
    trainer.plot_training_history(model_name=model.model_name)

    # 加载最佳模型进行评估
    print("加载最佳模型进行评估...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model_path = f'model/best_{model.model_name}_model.pth'
    model.load_state_dict(torch.load(best_model_path))

    # 评估模型
    eval_results = evaluate_model(model, val_loader, dataset, device, model_name=model.model_name)

    # 保存最终模型
    final_model_path = f'model/final_{model.model_name}_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"模型已保存为 {final_model_path}")


if __name__ == "__main__":
    main()
