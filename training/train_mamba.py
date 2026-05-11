"""
Semantic-Mamba 训练脚本 — 精简版
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
from sklearn.metrics import accuracy_score, f1_score

from model.semantic_mamba import SemanticMamba
from data_utils import RadarDataset


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, labels=target)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data, labels=target)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    return total_loss / len(loader), 100. * correct / total, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
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
    print(f"参数量: {total_params:.2f}M, 语义增强: {'开启' if not args.no_semantic else '关闭'}")

    # ---- 训练 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), PROJECT_ROOT / 'model' / 'best_semantic_mamba.pth')

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    elapsed = (time.time() - start) / 60
    print(f"\n训练完成! 耗时 {elapsed:.1f}min, 最佳验证Acc: {best_acc:.2f}%")

    # ---- 保存最终模型 ----
    torch.save(model.state_dict(), PROJECT_ROOT / 'model' / 'final_semantic_mamba.pth')
    print("模型已保存: model/final_semantic_mamba.pth")


if __name__ == '__main__':
    main()
