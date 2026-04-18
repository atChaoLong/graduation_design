import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def analyze_confusion(y_true, y_pred, label_names):
    """
    Analyze which labels are most confused with each other.

    Returns:
        confusion_pairs: list of (label1, label2, count) sorted by count desc
    """
    n = len(label_names)
    confusion_mat = np.zeros((n, n))

    for i in range(len(y_true)):
        true_idx = np.where(y_true[i] == 1)[0]
        pred_idx = np.where(y_pred[i] == 1)[0]
        for t in true_idx:
            for p in pred_idx:
                if t != p:
                    confusion_mat[t, p] += 1

    pairs = []
    for i in range(n):
        for j in range(n):
            if confusion_mat[i, j] > 0:
                pairs.append((label_names[i], label_names[j], int(confusion_mat[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs


def analyze_errors(y_true, y_pred, label2idx, raw_facts, orig_accusations, save_dir):
    """
    Analyze prediction errors and save results.

    Args:
        y_true: (N, L) binary ground truth
        y_pred: (N, L) binary predictions
        label2idx: dict mapping 'accusation::xxx' or 'article::xxx' to idx
        raw_facts: list of fact strings
        orig_accusations: list of original accusation lists
        save_dir: directory to save results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    idx2label = {v: k for k, v in label2idx.items()}

    # 1. Collect error samples
    error_samples = []
    for i in range(len(y_true)):
        if not np.array_equal(y_true[i], y_pred[i]):
            true_labels = [idx2label[idx].split('::', 1)[1] for idx in np.where(y_true[i] == 1)[0]]
            pred_labels = [idx2label[idx].split('::', 1)[1] for idx in np.where(y_pred[i] == 1)[0]]
            error_samples.append({
                'fact': raw_facts[i] if i < len(raw_facts) else '',
                'true_accusation': true_labels,
                'pred_accusation': pred_labels,
            })

    # Save error samples
    error_file = f"{save_dir}/error_samples.json"
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_samples[:100], f, ensure_ascii=False, indent=2)

    # 2. Confusion pairs (accusations only)
    acc_indices = [idx for label, idx in label2idx.items() if label.startswith('accusation::')]
    acc_names = [label.split('::', 1)[1] for label in label2idx.keys() if label.startswith('accusation::')]
    name_to_idx = {name: idx for name, idx in zip(acc_names, acc_indices)}

    acc_true = y_true[:, acc_indices]
    acc_pred = y_pred[:, acc_indices]

    confusion_pairs = analyze_confusion(acc_true, acc_pred, acc_names)

    # Save confusion pairs
    confusion_file = f"{save_dir}/confusion_pairs.json"
    with open(confusion_file, 'w', encoding='utf-8') as f:
        json.dump(confusion_pairs[:20], f, ensure_ascii=False, indent=2)

    # 3. Plot confusion heatmap
    n_acc = len(acc_names)
    if n_acc > 0 and n_acc <= 20:
        confusion_mat = np.zeros((n_acc, n_acc))
        for i in range(len(acc_true)):
            true_idx = np.where(acc_true[i] == 1)[0]
            pred_idx = np.where(acc_pred[i] == 1)[0]
            for t in true_idx:
                for p in pred_idx:
                    if t != p:
                        confusion_mat[t, p] += 1

        # Normalize by row
        row_sums = confusion_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        confusion_mat_norm = confusion_mat / row_sums

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(confusion_mat_norm, xticklabels=acc_names, yticklabels=acc_names,
                    cmap='Blues', ax=ax, annot=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Accusation Confusion Heatmap')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(f"{save_dir}/confusion_heatmap.png", dpi=150)
        plt.close(fig)

    return error_samples, confusion_pairs


def print_error_analysis(error_samples, confusion_pairs, top_n=10):
    """Print a summary of error analysis."""
    print("\n" + "=" * 60)
    print("错误分析报告")
    print("=" * 60)

    print(f"\n总错误样本数: {len(error_samples)}")

    print(f"\n--- Top {top_n} 混淆罪名对 ---")
    print(f"{'真实罪名':<12} {'预测罪名':<12} {'混淆次数':<10}")
    print("-" * 36)
    for true_lbl, pred_lbl, count in confusion_pairs[:top_n]:
        print(f"{true_lbl:<12} {pred_lbl:<12} {count:<10}")

    print(f"\n--- 错误样例 (前5条) ---")
    for i, sample in enumerate(error_samples[:5]):
        print(f"\n样例 {i + 1}:")
        print(f"  事实: {sample['fact'][:100]}...")
        print(f"  真实罪名: {sample['true_accusation']}")
        print(f"  预测罪名: {sample['pred_accusation']}")
