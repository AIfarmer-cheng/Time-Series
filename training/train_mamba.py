"""
Semantic-Mamba 训练脚本 — 正确版（无标签泄露）
===============================
用法:
    python training/train_mamba.py                      # 语义增强模式
    python training/train_mamba.py --no_semantic         # 纯雷达Mamba(消融)
    python training/train_mamba.py --epochs 60 --lr 0.0005
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import json
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from model.semantic_mamba import SemanticMamba
from data_utils import RadarDataset


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    all_probs = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            probs = torch.softmax(output, dim=1)
            _, pred = torch.max(probs, 1)

            correct += (pred == target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 基础指标
    acc = accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100

    # Top-3 / Top-5 准确率
    all_probs = np.array(all_probs)
    top3_acc = 0
    top5_acc = 0
    for i, true_label in enumerate(all_targets):
        top3 = np.argsort(all_probs[i])[-3:]
        top5 = np.argsort(all_probs[i])[-5:]
        if true_label in top3:
            top3_acc += 1
        if true_label in top5:
            top5_acc += 1
    top3_acc = top3_acc / len(all_targets) * 100
    top5_acc = top5_acc / len(all_targets) * 100

    return {
        'loss': total_loss / len(loader),
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'top3_acc': top3_acc,
        'top5_acc': top5_acc,
        'preds': all_preds,
        'targets': all_targets,
    }


def plot_training_curves(history, save_path):
    """绘制训练过程曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Process', fontsize=16)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax.plot(epochs, history['val_top3_acc'], 'g--', label='Val Top-3 Acc')
    ax.plot(epochs, history['val_top5_acc'], 'm--', label='Val Top-5 Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 Score
    ax = axes[0, 2]
    ax.plot(epochs, history['val_f1_macro'], 'r-', label='F1 Macro')
    ax.plot(epochs, history['val_f1_weighted'], 'orange', label='F1 Weighted')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('F1 Score Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall
    ax = axes[1, 0]
    ax.plot(epochs, history['val_precision'], 'c-', label='Precision')
    ax.plot(epochs, history['val_recall'], 'y-', label='Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score (%)')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], 'purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Confusion Matrix (最后一轮)
    ax = axes[1, 2]
    cm = confusion_matrix(history['final_targets'], history['final_preds'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix (Last Epoch)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def save_results(history, best_metrics, args, save_dir, model_tag):
    """保存训练结果和指标"""
    results = {
        'config': {
            'model': f'mamba_{model_tag}',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'd_model': args.d_model,
            'num_blocks': args.num_blocks,
            'use_semantic': not args.no_semantic,
        },
        'best_metrics': best_metrics,
        'history': {
            'train_loss': [round(x, 4) for x in history['train_loss']],
            'train_acc': [round(x, 2) for x in history['train_acc']],
            'val_loss': [round(x, 4) for x in history['val_loss']],
            'val_acc': [round(x, 2) for x in history['val_acc']],
            'val_f1_macro': [round(x, 2) for x in history['val_f1_macro']],
            'val_f1_weighted': [round(x, 2) for x in history['val_f1_weighted']],
            'val_precision': [round(x, 2) for x in history['val_precision']],
            'val_recall': [round(x, 2) for x in history['val_recall']],
            'val_top3_acc': [round(x, 2) for x in history['val_top3_acc']],
            'val_top5_acc': [round(x, 2) for x in history['val_top5_acc']],
            'lr': [round(x, 6) for x in history['lr']],
        }
    }

    save_path = save_dir / f'training_results_{model_tag}.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"训练结果已保存: {save_path}")

    # 保存详细分类报告
    if 'final_targets' in history and 'final_preds' in history:
        report = classification_report(
            history['final_targets'], history['final_preds'],
            output_dict=True, zero_division=0
        )
        report_path = save_dir / f'classification_report_{model_tag}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"分类报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--no_semantic', action='store_true',
                       help='关闭语义增强（纯雷达Mamba消融实验）')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # ---- 数据 ----
    data_dir = PROJECT_ROOT / "chinese_radar_dataset"
    dataset = RadarDataset(data_dir, 'labels.csv', 'characters.csv')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    print(f"类别数: {dataset.num_classes}, 训练: {len(train_set)}, 验证: {len(val_set)}")

    # ---- 模型 ----
    model = SemanticMamba(
        num_classes=dataset.num_classes,
        d_model=args.d_model,
        num_mamba_blocks=args.num_blocks,
        use_semantic=not args.no_semantic,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_tag = 'semantic' if not args.no_semantic else 'no_semantic'
    print(f"参数量: {total_params:.2f}M, 语义增强: {'开启' if not args.no_semantic else '关闭'}")

    # ---- 训练 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_macro': [], 'val_f1_weighted': [],
        'val_precision': [], 'val_recall': [],
        'val_top3_acc': [], 'val_top5_acc': [],
        'lr': [],
    }

    best_acc = 0
    best_metrics = {}
    start = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_results = validate(model, val_loader, criterion, device, dataset.num_classes)
        scheduler.step()

        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['acc'])
        history['val_f1_macro'].append(val_results['f1_macro'])
        history['val_f1_weighted'].append(val_results['f1_weighted'])
        history['val_precision'].append(val_results['precision_macro'])
        history['val_recall'].append(val_results['recall_macro'])
        history['val_top3_acc'].append(val_results['top3_acc'])
        history['val_top5_acc'].append(val_results['top5_acc'])
        history['lr'].append(scheduler.get_last_lr()[0])

        # 保存最后一轮的预测结果用于混淆矩阵
        history['final_preds'] = val_results['preds']
        history['final_targets'] = val_results['targets']

        # 更新最佳模型
        if val_results['acc'] > best_acc:
            best_acc = val_results['acc']
            best_metrics = {
                'epoch': epoch + 1,
                'val_acc': round(val_results['acc'], 2),
                'val_f1_macro': round(val_results['f1_macro'], 2),
                'val_f1_weighted': round(val_results['f1_weighted'], 2),
                'val_precision': round(val_results['precision_macro'], 2),
                'val_recall': round(val_results['recall_macro'], 2),
                'val_top3_acc': round(val_results['top3_acc'], 2),
                'val_top5_acc': round(val_results['top5_acc'], 2),
            }
            torch.save(model.state_dict(), PROJECT_ROOT / 'model' / f'best_{model_tag}_mamba.pth')

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_results['loss']:.4f} Acc: {val_results['acc']:.2f}% "
              f"F1: {val_results['f1_macro']:.2f}% Top-3: {val_results['top3_acc']:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    elapsed = (time.time() - start) / 60
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}")
    print(f"耗时: {elapsed:.1f}min")
    print(f"\n最佳验证指标 (Epoch {best_metrics['epoch']}):")
    print(f"  Accuracy:    {best_metrics['val_acc']:.2f}%")
    print(f"  F1 Macro:    {best_metrics['val_f1_macro']:.2f}%")
    print(f"  F1 Weighted: {best_metrics['val_f1_weighted']:.2f}%")
    print(f"  Precision:   {best_metrics['val_precision']:.2f}%")
    print(f"  Recall:      {best_metrics['val_recall']:.2f}%")
    print(f"  Top-3 Acc:   {best_metrics['val_top3_acc']:.2f}%")
    print(f"  Top-5 Acc:   {best_metrics['val_top5_acc']:.2f}%")

    # ---- 保存结果 ----
    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)

    # 保存训练曲线图
    plot_path = results_dir / f'training_curves_{model_tag}.png'
    plot_training_curves(history, plot_path)

    # 保存JSON结果
    save_results(history, best_metrics, args, results_dir, model_tag)

    # 保存最终模型
    final_path = PROJECT_ROOT / 'model' / f'final_{model_tag}_mamba.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n最终模型已保存: model/final_{model_tag}_mamba.pth")


if __name__ == '__main__':
    main()
